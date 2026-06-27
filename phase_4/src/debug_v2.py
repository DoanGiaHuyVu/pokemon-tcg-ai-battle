import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from cg.game import battle_start, battle_select
import agents.random_agent as random_agent
from src.nn.neural_v2_search_agent import agent as v2_agent
from src.nn.neural_agent_v2 import my_deck

def run_debug():
    seed = 42
    p1 = {"name": "v2", "agent": v2_agent, "deck": my_deck}
    p2 = {"name": "random", "agent": random_agent.agent, "deck": my_deck}
    
    obs_dict, _ = battle_start(p1["deck"], p2["deck"])
    
    step = 0
    while True:
        if obs_dict.get("current") and obs_dict["current"]["result"] != -1:
            print(f"Game over at step {step}")
            break
            
        if obs_dict.get("current")["yourIndex"] == 0:
            action = p1["agent"](obs_dict)
            print(f"Step {step} | P1 V2 Select: {obs_dict.get('select', {}).get('mode')} | Options: {len(obs_dict.get('select', {}).get('option',[]))} | Chosen: {action}")
            obs_dict = battle_select(action)
        else:
            action = p2["agent"](obs_dict)
            obs_dict = battle_select(action)
            
        step += 1
        
        if step > 500:
            print("Timeout!")
            break

if __name__ == "__main__":
    run_debug()
