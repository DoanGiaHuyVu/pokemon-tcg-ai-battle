import sys
import os
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from cg.game import battle_start, battle_select
import agents.dragapult_agent as dragapult_agent
from src.nn import neural_v3_search_agent

# Patch agent to print decisions
original_agent = neural_v3_search_agent.agent

def debug_agent(obs):
    print("--- TURN ---")
    if "select" in obs:
        opts = obs["select"].get("option", [])
        print("Options:")
        for i, opt in enumerate(opts):
            print(f"  {i}: {opt}")
    
    action = original_agent(obs)
    print(">>> CHOSE:", action)
    return action

neural_v3_search_agent.agent = debug_agent
neural_v3_search_agent.STRICT_SEARCH = False

print("Starting debug match...")
try:
    obs, _ = battle_start(dragapult_agent.my_deck, dragapult_agent.my_deck)
    step = 0
    while True:
        if obs["current"].get("result", -1) != -1:
            print("GAME OVER. Winner:", obs["current"]["result"])
            break
        
        yourIndex = obs["current"]["yourIndex"]
        if yourIndex == 0:
            action = debug_agent(obs)
        else:
            action = dragapult_agent.agent(obs)
            
        obs = battle_select(action)
        step += 1
        if step > 200:
            print("MAX STEPS")
            break
except Exception as e:
    print("CRASH:", e)
