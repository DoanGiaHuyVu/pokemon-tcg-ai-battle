import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.dirname(BASE_DIR))
from cg.game import battle_start, battle_select
from cg.api import search_begin, search_step, search_release

import agents.dragapult_agent as dragapult

def run_probe():
    print("Starting probe...")
    obs, _ = battle_start(dragapult.my_deck, dragapult.my_deck)
    
    import agents.dragapult_agent as p1_agent
    import agents.dragapult_agent as p2_agent
    p1_agent.my_deck = dragapult.my_deck.copy()
    p2_agent.my_deck = dragapult.my_deck.copy()
    
    # Run 10 moves
    step = 0
    while step < 10:
        if obs.get("current", {}).get("yourIndex", 0) == 0:
            action = p1_agent.agent(obs)
        else:
            action = p2_agent.agent(obs)
            
        obs = battle_select(action)
        step += 1
        
    print("Found normal action state after 10 steps")
    
    # We are in a state where we can act.
    state = obs.get("current", {})
    p1 = state.get("players", [{}, {}])[1 - state.get("yourIndex", 0)]
    print("p1 handCount:", p1.get("handCount"))
    print("p1 deckCount:", p1.get("deckCount"))
    print("p1 prize:", p1.get("prize"))
    print("p1 active:", p1.get("active"))
    print("p1 bench:", p1.get("bench"))
    
    opp_deck = dragapult.my_deck.copy()
    n_prize = len(p1.get("prize") or [0]*6)
    opp_prize = opp_deck[:n_prize]
    opp_deck = opp_deck[n_prize:]
    
    n_hand = len(p1.get("hand") or [0]*7)
    opp_hand = opp_deck[:n_hand]
    opp_deck = opp_deck[n_hand:]
    
    opp_active = p1.get("active")
    if not opp_active:
        # If no active yet, just provide an empty array? No, the game expects a card.
        opp_active = opp_deck[:1]
        opp_deck = opp_deck[1:]
    else:
        # It's already a list of dictionaries with ids, or maybe just a list of integers?
        if isinstance(opp_active[0], dict):
            opp_active = [x["id"] for x in opp_active]
        else:
            opp_active = opp_active
        
    try:
        from cg.api import to_observation_class
        agent_obs = to_observation_class(obs)
        search_state = search_begin(
            agent_observation=agent_obs,
            your_deck=dragapult.my_deck,
            your_prize=dragapult.my_deck[:6],
            opponent_deck=opp_deck,
            opponent_hand=opp_hand,
            opponent_prize=opp_prize,
            opponent_active=opp_active,
            manual_coin=0
        )
        print("search_begin Success!")
        print("search_state type:", type(search_state))
        print("search_state dir:", [x for x in dir(search_state) if not x.startswith("__")])
        print("search_state repr:", repr(search_state))
        
        search_id = search_state.searchId
        
        # Now probe search step
        options = obs.get("select", {}).get("option", [])
        action = [0]
        
        next_state = search_step(search_id, action)
        print("search_step Success!")
        print("next_state type:", type(next_state))
        print("next_state dir:", [x for x in dir(next_state) if not x.startswith("__")])
        print("next_state repr:", repr(next_state))
        
        import dataclasses
        obs_dict = dataclasses.asdict(next_state.observation)
        print("obs_dict type:", type(obs_dict))
        print("obs_dict keys:", obs_dict.keys())
        
        search_release(search_id)
        search_release(next_state.searchId)
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_probe()
