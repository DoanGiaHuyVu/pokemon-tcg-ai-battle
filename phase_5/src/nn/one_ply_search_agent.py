import os
import sys
import copy
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from cg.api import search_begin, search_step, search_end, search_release
from src.nn.tactical_evaluator import score_state
import agents.dragapult_agent as fallback_agent

def get_legal_actions(obs):
    """Returns a list of legal actions (indices) for the current state."""
    # If the simulator provided select.option, we just use their lengths.
    options = obs.get("select", {}).get("option", [])
    if not options:
        return [[0]]
        
    # We will generate a naive list of legal actions. 
    # For now, we only explore single-selections (minCount=1, maxCount=1) to keep branching factor low.
    min_count = obs["select"].get("minCount", 1)
    max_count = obs["select"].get("maxCount", 1)
    
    actions = []
    if min_count == 0:
        actions.append([]) # Optional action (Pass)
        
    for i in range(len(options)):
        actions.append([i])
        
    return actions

def get_unseen_cards(my_deck_list, known_cards):
    """Simple determinization logic."""
    unseen = list(my_deck_list)
    for c in known_cards:
        if c in unseen:
            unseen.remove(c)
    return unseen

def agent(obs: dict) -> list[int]:
    """
    1-Ply Tactical Search Agent.
    """
    if "select" not in obs or obs["select"] is None:
        return [0]
        
    # Only search if maxCount <= 1 to avoid combinatorial explosion for now
    if obs["select"].get("maxCount", 1) > 1:
        return fallback_agent.agent(obs)
        
    legal_actions = get_legal_actions(obs)
    
    # If there's only 1 legal action (e.g., forced), just return it
    if len(legal_actions) == 1:
        return legal_actions[0]

    # --- Determinize hidden state ---
    # We cheat slightly for phase 4 by using the known opponent deck (Dragapult vs Dragapult)
    opp_deck_list = fallback_agent.my_deck
    opp_hand = [] # We could randomly sample from opp_deck_list
    opp_deck = opp_deck_list.copy()
    opp_prize = opp_deck_list[:6] # Very rough cheat
    opp_active = [] # We don't guess facedown active yet
    
    # Create an AgentObservation mock
    # Wait, the python cg.api requires `agent_observation` which is a dataclass.
    # It's easier to use the parsed `obs` dict if we convert it back, but `api.py` has `to_observation_class`!
    from cg.api import to_observation_class
    try:
        agent_obs = to_observation_class(obs)
        search_state = search_begin(
            agent_observation=agent_obs,
            your_deck=fallback_agent.my_deck, # Approximation
            your_prize=fallback_agent.my_deck[:6],
            opponent_deck=opp_deck,
            opponent_prize=opp_prize,
            opponent_hand=opp_hand,
            opponent_active=opp_active,
            manual_coin=0
        )
    except Exception as e:
        # Failed to begin search (likely invalid determinization). Fallback.
        return fallback_agent.agent(obs)

    best_action = fallback_agent.agent(obs) # Default to heuristic
    best_score = -999999.0
    
    # --- Search ---
    search_id = search_state.searchId
    
    for action in legal_actions:
        try:
            next_state = search_step(search_id, action)
            score = score_state(next_state.state, agent_obs.current.yourIndex)
            
            if score > best_score:
                best_score = score
                best_action = action
                
        except Exception as e:
            # Action was illegal in this deterministic world
            pass
            
    # Cleanup search memory
    search_release(search_id)
    
    return best_action
