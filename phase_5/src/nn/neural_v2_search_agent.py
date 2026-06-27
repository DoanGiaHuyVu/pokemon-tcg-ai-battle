import os
import sys
import copy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from cg.api import search_begin, search_step, search_end, search_release, to_observation_class
from src.nn.tactical_evaluator import score_state
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

def agent(obs: dict) -> list[int] | int:
    """
    Neural v2 1-Ply Search Agent.
    Evaluates actions using a combination of Neural Policy Prior and Tactical Search Post-Evaluation.
    """
    metrics = obs.get("metrics")
    if metrics is None:
        print("WARNING: METRICS IS NONE IN AGENT!")
    
    if "select" not in obs or obs["select"] is None:
        return [0]
        
    if obs["select"].get("maxCount", 1) > 1:
        if metrics: metrics.log_fallback("maxCount > 1")
        return fallback_agent.agent(obs)
        
    legal_actions = get_legal_actions(obs)
    
    if len(legal_actions) == 1:
        return legal_actions[0]

    policy_scores = get_neural_policy_scores(obs)
    
    opp_deck = fallback_agent.my_deck.copy()
    # Safely mock opponent hidden state based on known counts
    import copy
    state = obs.get("current", {})
    if state:
        p1 = state.get("players", [{}, {}])[1 - state.get("yourIndex", 0)]
        
        n_prize = len(p1.get("prize") or [0]*6)
        opp_prize = opp_deck[:n_prize]
        opp_deck = opp_deck[n_prize:]
        
        n_hand = len(p1.get("hand") or [0]*7)
        opp_hand = opp_deck[:n_hand]
        opp_deck = opp_deck[n_hand:]
    else:
        opp_prize = opp_deck[:6]
        opp_hand = opp_deck[6:13]
        opp_deck = opp_deck[13:]
        
    opp_active = []
    
    try:
        agent_obs = to_observation_class(obs)
        search_state = search_begin(
            agent_observation=agent_obs,
            your_deck=my_deck, 
            your_prize=my_deck[:6],
            opponent_deck=opp_deck,
            opponent_prize=opp_prize,
            opponent_hand=opp_hand,
            opponent_active=opp_active,
            manual_coin=0
        )
    except Exception as e:
        if metrics: metrics.log_search_exception(f"search_begin: {str(e)}")
        if STRICT_SEARCH:
            raise RuntimeError(f"Strict Search Exception: {e}")
        # Fallback to pure policy if search init fails
        from src.nn.neural_agent_v2 import agent as fallback_v2
        return fallback_v2(obs)

    best_action = fallback_agent.agent(obs) 
    best_score = -999999.0
    search_id = search_state.searchId
    
    for action in legal_actions:
        if metrics: metrics.log_search_attempt()
        try:
            next_state = search_step(search_id, action)
            if metrics: metrics.log_search_success()
            tactical_score = score_state(next_state.state, agent_obs.current.yourIndex)
            search_release(next_state.searchId)
        except Exception as e:
            if metrics: metrics.log_search_exception(f"search_step: {str(e)}")
            if STRICT_SEARCH:
                raise RuntimeError(f"Strict Search Exception: {e}")
            continue
            
        # Combine Tactical Score with Neural Prior Score
        prior_score = policy_scores.get(tuple(action), 0.0)
        
        # Weighted combo: 1.0 * Tactical + 100.0 * NeuralPrior
        combined_score = tactical_score + (100.0 * prior_score)
        
        if combined_score > best_score:
            best_score = combined_score
            best_action = action
            
    search_release(search_id)
    
    if metrics: metrics.log_decision_source(is_neural=True)
    return best_action
