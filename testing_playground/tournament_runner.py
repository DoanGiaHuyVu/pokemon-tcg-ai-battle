import json
import traceback
import time
import os
import sys
import importlib

from cg.game import battle_start, battle_select, battle_finish

def run_match(agent_0_name, agent_1_name):
    # Load agents
    mod_0 = importlib.import_module(f"agents.{agent_0_name}_agent")
    mod_1 = importlib.import_module(f"agents.{agent_1_name}_agent")
    
    agent_0 = mod_0.agent
    agent_1 = mod_1.agent
    
    try:
        deck_0 = mod_0.my_deck
    except AttributeError:
        deck_0 = mod_0.read_deck_csv()
        
    try:
        deck_1 = mod_1.my_deck
    except AttributeError:
        deck_1 = mod_1.read_deck_csv()

    print(f"Starting match: {agent_0_name} vs {agent_1_name}...", end=" ")
    
    obs_dict, start_data = battle_start(deck_0, deck_1)
    
    if obs_dict is None:
        print("Failed to start battle.")
        return -1
        
    step_count = 0
    start_time = time.time()
    
    try:
        while True:
            if obs_dict.get("current") and obs_dict["current"]["result"] != -1:
                winner = obs_dict["current"]["result"]
                winner_name = agent_0_name if winner == 0 else agent_1_name
                print(f"Winner: {winner} ({winner_name}) in {step_count} steps.")
                return winner
                
            your_index = obs_dict["current"]["yourIndex"]
            agent = agent_0 if your_index == 0 else agent_1
            
            action = agent(obs_dict)
                
            obs_dict = battle_select(action)
            step_count += 1
            
            if step_count > 5000:
                print("Max steps exceeded. Draw.")
                return -1
                
    except Exception as e:
        print(f"Crash on step {step_count}: {e}")
        return -1

if __name__ == "__main__":
    agents = ["dragapult", "mega_abomasnow", "mega_lucario", "iono", "random"]
    
    print("--- VS Random Agent ---")
    for a in agents[:-1]:
        run_match(a, "random")
        
    print("\n--- Round Robin (Starter Agents) ---")
    for i in range(len(agents)-1):
        for j in range(i+1, len(agents)-1):
            run_match(agents[i], agents[j])
