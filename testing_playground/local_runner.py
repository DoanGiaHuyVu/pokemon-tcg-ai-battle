import json
import traceback
import time
import os

from cg.game import battle_start, battle_select, battle_finish
from dragapult_heuristic import agent as agent_0
from random_agent import agent as agent_1
from dragapult_heuristic import my_deck as deck_0
from random_agent import read_deck_csv

def run_match():
    deck_1 = read_deck_csv()
    
    agents = [agent_0, agent_1]
    
    print("Starting match: Dragapult Heuristic vs Random Agent")
    
    obs_dict, start_data = battle_start(deck_0, deck_1)
    
    if obs_dict is None:
        print("Failed to start battle.")
        return
        
    match_history = []
    step_count = 0
    start_time = time.time()
    
    try:
        while True:
            # Check if match is finished
            if obs_dict.get("current") and obs_dict["current"]["result"] != -1:
                winner = obs_dict["current"]["result"]
                print(f"Match finished in {step_count} steps. Winner: {winner}")
                break
                
            your_index = obs_dict["current"]["yourIndex"]
            agent = agents[your_index]
            
            # Step agent
            action = agent(obs_dict)
            
            # Track history for debugging crashes
            match_history.append({
                "step": step_count,
                "player": your_index,
                "obs": obs_dict,
                "action": action
            })
            
            obs_dict = battle_select(action)
            step_count += 1
            
            # Basic timeout/infinite loop prevention for local testing
            if time.time() - start_time > 60:
                raise TimeoutError("Match exceeded 60 seconds wall-clock time.")
                
    except Exception as e:
        print(f"Match ended with an error: {e}")
        traceback.print_exc()
        # Save replay on crash
        replay_path = os.path.join(os.path.dirname(__file__), f"crash_replay_{int(time.time())}.jsonl")
        with open(replay_path, "w") as f:
            for step in match_history:
                f.write(json.dumps(step) + "\n")
        print(f"Replay saved to {replay_path}")
        
    finally:
        battle_finish()

if __name__ == "__main__":
    run_match()
