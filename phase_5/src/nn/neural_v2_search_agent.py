import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from cg.api import search_begin, search_step, search_end, search_release, to_observation_class, all_card_data
from src.nn.tactical_evaluator import score_state, score_action_delta

# Pre-load card metadata for Basic Pokémon identification
_basic_pokemon_ids = None

def _get_basic_pokemon_ids():
    global _basic_pokemon_ids
    if _basic_pokemon_ids is None:
        cards = all_card_data()
        _basic_pokemon_ids = set(c.cardId for c in cards if c.basic)
    return _basic_pokemon_ids
from src.nn.neural_agent_v2 import model, parser, my_deck
import agents.dragapult_agent as fallback_agent
import torch

def get_legal_actions(obs):
    options = obs.get("select", {}).get("option", [])
    if not options:
        return [[0]]
        
    min_count = obs["select"].get("minCount", 1)
    
    actions = []
    if min_count == 0:
        actions.append([]) 
        
    for i in range(len(options)):
        actions.append([i])
        
    return actions

def get_neural_policy_scores(obs):
    """Returns a dict of {action_tuple: score} from the neural policy."""
    options = obs.get("select", {}).get("option", [])
    if not options:
        return {}
        
    try:
        from src.nn.neural_agent_v2 import device
        your_index = obs.get("current", {}).get("yourIndex", 0)
        global_features = parser._parse_global_features(obs.get("current")).unsqueeze(0).to(device)
        card_tokens, numeric_features, zone_masks = parser._parse_cards(obs.get("current"), your_index)
        deck_id = torch.tensor([0], dtype=torch.long).to(device)
        
        state_inputs = {
            "deck_id": deck_id,
            "card_tokens": card_tokens.unsqueeze(0).to(device),
            "numeric_features": numeric_features.unsqueeze(0).to(device),
            "global_features": global_features,
            "zone_masks": zone_masks.unsqueeze(0).to(device)
        }
        
        action_inputs = parser._parse_actions(options)
        for k in action_inputs:
            action_inputs[k] = action_inputs[k].unsqueeze(0).to(device)
            
        with torch.no_grad():
            logits, _, _ = model(state_inputs, action_inputs)
            
        # Apply softmax to keep prior bounded between [0, 1]
        probs = torch.softmax(logits[0], dim=-1).tolist()
        mask = action_inputs["action_mask"][0].tolist()
        
        policy_scores = {}
        for i, (score, m) in enumerate(zip(probs, mask)):
            if m > 0.5:
                policy_scores[(i,)] = score
                
        # Assign a neutral score to PASS action if minCount == 0
        if obs["select"].get("minCount", 1) == 0:
            policy_scores[()] = 0.0 # Baseline pass score
            
        return policy_scores
    except Exception as e:
        print(f"Policy error: {e}")
        return {}

STRICT_SEARCH = False

def build_determinization(obs):
    """
    Build determinized opponent hidden state by:
    1. Reading exact counts from the simulator (deckCount, handCount, len(prize))
    2. Collecting all known public opponent cards (active, bench, discard, energy, tools)
    3. Subtracting known cards from the full decklist to form the unknown pool
    4. Sampling exact counts from the unknown pool for hand, prize, and deck
    """
    state = obs.get("current", {})
    if not state:
        return None, None, None, None
    
    your_index = state.get("yourIndex", 0)
    opp_index = 1 - your_index
    opp = state.get("players", [{}, {}])[opp_index]
    
    # Read exact counts the simulator expects
    true_hand_count = opp.get("handCount", 0) or 0
    true_deck_count = opp.get("deckCount", 0) or 0
    true_prize_count = len(opp.get("prize") or [])
    
    # Collect all publicly known opponent card IDs
    known_public_ids = []
    
    # Active pokemon and their attached energy/tools
    for pkmn in (opp.get("active") or []):
        if isinstance(pkmn, dict):
            if pkmn.get("id") is not None:
                known_public_ids.append(pkmn["id"])
            for e_card in (pkmn.get("energyCards") or []):
                if isinstance(e_card, dict) and e_card.get("id") is not None:
                    known_public_ids.append(e_card["id"])
            for tool in (pkmn.get("tools") or []):
                if isinstance(tool, dict) and tool.get("id") is not None:
                    known_public_ids.append(tool["id"])
            for pre_evo in (pkmn.get("preEvolution") or []):
                if isinstance(pre_evo, dict) and pre_evo.get("id") is not None:
                    known_public_ids.append(pre_evo["id"])
    
    # Bench pokemon and their attached energy/tools
    for pkmn in (opp.get("bench") or []):
        if isinstance(pkmn, dict):
            if pkmn.get("id") is not None:
                known_public_ids.append(pkmn["id"])
            for e_card in (pkmn.get("energyCards") or []):
                if isinstance(e_card, dict) and e_card.get("id") is not None:
                    known_public_ids.append(e_card["id"])
            for tool in (pkmn.get("tools") or []):
                if isinstance(tool, dict) and tool.get("id") is not None:
                    known_public_ids.append(tool["id"])
            for pre_evo in (pkmn.get("preEvolution") or []):
                if isinstance(pre_evo, dict) and pre_evo.get("id") is not None:
                    known_public_ids.append(pre_evo["id"])
    
    # Discard pile
    for card in (opp.get("discard") or []):
        if isinstance(card, dict) and card.get("id") is not None:
            known_public_ids.append(card["id"])
    
    # Build unknown pool: start from the full decklist and remove known public cards
    unknown_pool = fallback_agent.my_deck.copy()
    for cid in known_public_ids:
        if cid in unknown_pool:
            unknown_pool.remove(cid)
    
    # Sample exact counts from the unknown pool
    needed = true_hand_count + true_prize_count + true_deck_count
    if needed > len(unknown_pool):
        # Not enough cards in pool — determinization is impossible
        return None, None, None, None
    
    opp_hand = unknown_pool[:true_hand_count]
    remaining = unknown_pool[true_hand_count:]
    
    opp_prize = remaining[:true_prize_count]
    remaining = remaining[true_prize_count:]
    
    opp_deck = remaining[:true_deck_count]
    
    # Determine opponent active prediction
    # Only needed if the active slot has a face-down (None) card
    opp_active_raw = opp.get("active") or []
    needs_active_prediction = (len(opp_active_raw) > 0 and opp_active_raw[0] is None)
    
    opp_active = []
    if needs_active_prediction:
        # Find a Basic Pokémon card ID from the unknown pool
        basic_ids = _get_basic_pokemon_ids()
        for cid in unknown_pool:
            if cid in basic_ids:
                opp_active = [cid]
                break
        if not opp_active:
            # No basic pokemon found in pool — determinization cannot proceed
            return None, None, None, None
    
    return opp_hand, opp_prize, opp_deck, opp_active


def agent(obs: dict) -> list[int] | int:
    """
    Neural v2 Turn-Level Beam Search Agent.
    Evaluates multi-action sequences within a turn using beam search,
    combined with Neural Policy Prior and Tactical Delta evaluation.
    """
    metrics = obs.get("metrics")
    
    if "select" not in obs or obs["select"] is None:
        return [0]
        
    if obs["select"].get("maxCount", 1) > 1:
        if metrics:
            metrics.log_multi_select_skip()
        return fallback_agent.agent(obs)
        
    legal_actions = get_legal_actions(obs)
    
    if len(legal_actions) == 1:
        if metrics:
            metrics.log_trivial_decision()
        return legal_actions[0]

    policy_scores = get_neural_policy_scores(obs)
    
    # Build determinization from public information
    opp_hand, opp_prize, opp_deck, opp_active = build_determinization(obs)
    
    if opp_hand is None:
        # Determinization failed — not enough cards in pool
        if metrics:
            metrics.log_fallback_decision("determinization_pool_exhausted")
        if STRICT_SEARCH:
            raise RuntimeError("Strict Search: determinization pool exhausted")
        from src.nn.neural_agent_v2 import agent as fallback_v2
        return fallback_v2(obs)
    
    # --- search_begin ---
    if metrics:
        metrics.log_begin_attempt()
    
    try:
        agent_obs = to_observation_class(obs)
        your_index = agent_obs.current.yourIndex
        
        # Build your own hidden state prediction
        my_state = obs["current"]["players"][your_index]
        my_deck_count = my_state.get("deckCount", 0) or 0
        my_prize_count = len(my_state.get("prize") or [])
        
        # For your own deck/prize, use your known cards from your hand etc.
        your_deck_pred = my_deck[:my_deck_count]
        your_prize_pred = my_deck[:my_prize_count]
        
        search_state = search_begin(
            agent_observation=agent_obs,
            your_deck=your_deck_pred, 
            your_prize=your_prize_pred,
            opponent_deck=opp_deck,
            opponent_prize=opp_prize,
            opponent_hand=opp_hand,
            opponent_active=opp_active,
            manual_coin=0
        )
        if metrics:
            metrics.log_begin_success()
    except Exception as e:
        if metrics:
            metrics.log_begin_failure(str(e))
            metrics.log_fallback_decision(f"search_begin: {e}")
        if STRICT_SEARCH:
            raise RuntimeError(f"Strict Search: search_begin failed: {e}")
        from src.nn.neural_agent_v2 import agent as fallback_v2
        return fallback_v2(obs)

    # --- Turn-Level Beam Search ---
    BEAM_WIDTH = 3
    MAX_DEPTH = 8
    PRIOR_WEIGHT = float(os.environ.get("PRIOR_WEIGHT", "10.0"))

    current_state = agent_obs.current
    root_search_id = search_state.searchId
    
    if metrics:
        metrics.log_legal_actions(len(legal_actions))

    # Each beam entry: (score, first_action, search_id, depth, is_our_turn)
    # We track the first_action so we know which root action led to this state
    best_action = fallback_agent.agent(obs)
    best_score = -999999.0
    any_eval_succeeded = False

    # ids_to_release tracks all search IDs we create so we can clean up
    ids_to_release = []

    # --- Depth 1: Expand root into all legal actions ---
    beam = []
    for action in legal_actions:
        if metrics:
            metrics.log_step_attempt()
        try:
            next_ss = search_step(root_search_id, action)
            if metrics:
                metrics.log_step_success()
        except Exception as e:
            if metrics:
                metrics.log_step_failure(str(e))
            if STRICT_SEARCH:
                search_release(root_search_id)
                raise RuntimeError(f"Strict Search: search_step failed: {e}")
            continue

        try:
            result_state = next_ss.observation.current
            tactical_delta = score_action_delta(current_state, result_state, your_index)
            if metrics:
                metrics.log_action_eval_success()
            any_eval_succeeded = True
        except Exception as e:
            if metrics:
                metrics.log_action_eval_failure()
            search_release(next_ss.searchId)
            continue

        prior_score = policy_scores.get(tuple(action), 0.0)
        combined = tactical_delta + (PRIOR_WEIGHT * prior_score)

        # Check if turn has ended (opponent's turn now)
        turn_still_ours = (result_state.yourIndex == your_index) if hasattr(result_state, 'yourIndex') else True

        if turn_still_ours:
            # Can explore deeper — add to beam
            beam.append((combined, action, next_ss.searchId, 1, result_state))
        else:
            # Turn ended — this is a terminal leaf, score it
            ids_to_release.append(next_ss.searchId)
            if combined > best_score:
                best_score = combined
                best_action = action

    # --- Depths 2..MAX_DEPTH: Beam expansion ---
    for depth in range(2, MAX_DEPTH + 1):
        if not beam:
            break

        # Keep only top-K beams
        beam.sort(key=lambda x: x[0], reverse=True)
        survivors = beam[:BEAM_WIDTH]
        # Release non-surviving search IDs
        for entry in beam[BEAM_WIDTH:]:
            ids_to_release.append(entry[2])
        beam = []

        for parent_score, first_action, parent_sid, parent_depth, parent_result_state in survivors:
            # Get the select data from this search state to find legal actions
            # We use search_step with each possible action index
            # The observation should have select data telling us what options exist
            parent_obs = None
            try:
                parent_obs = parent_result_state
            except Exception:
                ids_to_release.append(parent_sid)
                continue

            # We don't have the select options from the search state directly,
            # so we try action indices [0..9] and see which succeed
            expanded = False
            for action_idx in range(10):
                child_action = [action_idx]
                try:
                    child_ss = search_step(parent_sid, child_action)
                except Exception:
                    # This action index is not valid — stop trying higher indices
                    break

                try:
                    child_result = child_ss.observation.current
                    child_delta = score_action_delta(current_state, child_result, your_index)
                except Exception:
                    search_release(child_ss.searchId)
                    continue

                child_score = child_delta  # Deeper levels don't use prior (no obs available)
                turn_still_ours = (child_result.yourIndex == your_index) if hasattr(child_result, 'yourIndex') else True
                expanded = True

                if turn_still_ours and depth < MAX_DEPTH:
                    beam.append((child_score, first_action, child_ss.searchId, depth, child_result))
                else:
                    ids_to_release.append(child_ss.searchId)
                    if child_score > best_score:
                        best_score = child_score
                        best_action = first_action

            if not expanded:
                # No valid children — treat parent as leaf
                if parent_score > best_score:
                    best_score = parent_score
                    best_action = first_action
                ids_to_release.append(parent_sid)
            # If expanded, the parent_sid is consumed by beam entries or released above

    # Release any remaining beam entries that weren't expanded
    for entry in beam:
        if entry[0] > best_score:
            best_score = entry[0]
            best_action = entry[1]
        ids_to_release.append(entry[2])

    # Clean up all search IDs
    for sid in ids_to_release:
        try:
            search_release(sid)
        except Exception:
            pass
    search_release(root_search_id)
    
    if any_eval_succeeded:
        if metrics:
            metrics.log_search_decision()
    else:
        if metrics:
            metrics.log_fallback_decision("all_search_steps_failed")
    
    return best_action
