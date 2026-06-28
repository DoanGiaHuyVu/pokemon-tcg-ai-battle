#!/usr/bin/env python3
"""
Phase 7: Loss Analyzer v2 (First Critical Mistake Diagnosis)
Simulates games between beam_search and heuristic opponents.
For every loss, it analyzes the trajectory to find the "first critical mistake",
such as a missed KO or a turn ended with unattached energy.
"""
import argparse
import time
import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from cg.game import battle_start, battle_select
from cg.api import to_observation_class

import agents.dragapult_agent as dragapult_agent
from src.nn import neural_v3_search_agent
from src.nn.tactical_evaluator import _best_available_attack_damage, _get_active_hp_info, _can_active_attack, _min_shortfall_active

def analyze_losses(opponent_name, num_games):
    # Disable strict search so we don't crash on invalid simulated paths
    neural_v3_search_agent.STRICT_SEARCH = False
    
    wins = 0
    losses = 0
    
    mistake_counts = {
        "MISSED_KO": 0,
        "MISSED_ATTACK": 0,
        "TURN_ENDED_WITH_ENERGY_IN_HAND": 0,
        "OPPONENT_TOOK_EARLY_LEAD": 0,
    }
    
    print(f"\nStarting Loss Analyzer v2: beam_search vs {opponent_name} ({num_games} games)")
    
    for game_idx in range(num_games):
        try:
            obs_dict, _ = battle_start(dragapult_agent.my_deck, dragapult_agent.my_deck)
        except Exception as e:
            continue
            
        step_count = 0
        first_mistake = None
        last_opts = []
        last_action = []
        while True:
            if obs_dict.get("current") and obs_dict["current"]["result"] != -1:
                res = obs_dict["current"]["result"]
                if res == 1:
                    losses += 1
                    if first_mistake:
                        mistake_counts[first_mistake] += 1
                    print(f"Game {game_idx} LOST. First Mistake: {first_mistake}")
                else:
                    wins += 1
                break
                
            your_index = obs_dict["current"]["yourIndex"]
            
            if your_index == 0 and first_mistake is None:
                # We are beam search, let's analyze the state BEFORE we take action
                agent_obs = to_observation_class(obs_dict)
                my_state = agent_obs.current.players[0]
                opp_state = agent_obs.current.players[1]
                
                # Check for available KOs
                our_dmg = _best_available_attack_damage(agent_obs.current.players[0])
                opp_info = _get_active_hp_info(agent_obs.current.players[1])
                can_ko = False
                if opp_info and our_dmg > 0 and opp_info[0] <= our_dmg:
                    can_ko = True
                    
                can_attack = _can_active_attack(agent_obs.current.players[0])
                
                # Take action
                try:
                    if obs_dict.get("select") and obs_dict["select"].get("option"):
                        opts = obs_dict['select']['option']
                        last_opts = opts
                        print(f"[DEBUG] Available options: {len(opts)}")
                        for i, opt in enumerate(opts):
                            print(f"[DEBUG] Option {i}: {opt}")
                    action = neural_v3_search_agent.agent(obs_dict)
                    last_action = action
                    if last_opts and last_action and len(last_action) > 0:
                        opt_idx = last_action[0]
                        if opt_idx < len(last_opts) and last_opts[opt_idx].get('type') == 13:
                            print(f"\n[>>>] EXECUTING ATTACK! {last_opts[opt_idx]}")
                    
                    next_obs = battle_select(action)
                except Exception:
                    next_obs = obs_dict # Crash, will end loop
                    
                # Analyze state AFTER action
                if next_obs.get("current") and "yourIndex" in next_obs["current"]:
                    next_your_index = next_obs["current"]["yourIndex"]
                    turn_ended = (next_your_index != 0)
                    
                    if turn_ended:
                        # We ended our turn. Did we make a mistake?
                        if can_ko:
                            # We could have KO'd but we ended the turn
                            # Check if the opponent is still alive
                            agent_next = to_observation_class(next_obs)
                            opp_next_info = _get_active_hp_info(agent_next.current.players[1])
                            if opp_next_info and opp_next_info[0] > 0:
                                first_mistake = "MISSED_KO"
                        elif can_attack and not first_mistake:
                            # check if the action we took was actually an attack
                            action_was_attack = False
                            if last_opts and last_action and len(last_action) > 0:
                                opt_idx = last_action[0]
                                if opt_idx < len(last_opts) and last_opts[opt_idx].get('type') == 13:
                                    action_was_attack = True
                            
                            if not action_was_attack:
                                first_mistake = "MISSED_ATTACK"
                                print("\n[DEBUG] MISSED ATTACK DETECTED!")
                                print(f"Action taken: {last_action}")
                        elif not first_mistake:
                            # Check if we still have energy in hand and attackers missing energy
                            hand = my_state.hand or []
                            energy_in_hand = any(getattr(c, "cardType", 0) in [5, 6] for c in hand) # 5=BASIC, 6=SPECIAL
                            shortfall = _min_shortfall_active(my_state)
                            if energy_in_hand and shortfall > 0:
                                first_mistake = "TURN_ENDED_WITH_ENERGY_IN_HAND"
                            
                        # Check prize lead
                        if not first_mistake:
                            my_prizes = len(my_state.prize or [])
                            opp_prizes = len(opp_state.prize or [])
                            if opp_prizes < my_prizes and opp_prizes <= 4:
                                first_mistake = "OPPONENT_TOOK_EARLY_LEAD"
                
                obs_dict = next_obs
            else:
                # Heuristic turn
                action = dragapult_agent.agent(obs_dict)
                try:
                    obs_dict = battle_select(action)
                except Exception:
                    break
                    
            step_count += 1
            if step_count > 1000:
                break
                
    print(f"\n{'='*50}")
    print(f"LOSS ANALYSIS RESULTS (Over {losses} losses)")
    print(f"{'='*50}")
    for k, v in mistake_counts.items():
        pct = (v / losses * 100) if losses > 0 else 0
        print(f"{k:<35}: {v:>3} ({pct:.1f}%)")
    print(f"{'='*50}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=50)
    args = parser.parse_args()
    analyze_losses("dragapult", args.games)
