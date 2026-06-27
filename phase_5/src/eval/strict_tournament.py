import argparse
import time
import json
import statistics
import os
import sys

# Ensure cg is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from cg.game import battle_start, battle_select
import agents.random_agent as random_agent
import agents.dragapult_agent as heuristic_agent
neural_deck = heuristic_agent.my_deck

from src.eval.metrics import MetricsRegistry

class AgentWrapper:
    def __init__(self, name, agent_func, deck, is_neural=False, use_fallback=True):
        self.name = name
        self.agent_func = agent_func
        self.deck = deck
        self.is_neural = is_neural
        self.use_fallback = use_fallback
        
        # Stats
        self.decision_times = []
        self.fallback_count = 0
        self.multi_select_count = 0
        self.total_decisions = 0
        self.illegal_actions = 0
        self.metrics = MetricsRegistry()
        
    def select_action(self, obs):
        start_t = time.perf_counter()
        
        # Track multi-select
        select_block = obs.get("select") or {}
        options = select_block.get("option", [])
        max_count = select_block.get("maxCount", 1)
        
        if max_count > 1:
            self.multi_select_count += 1
            
        obs["metrics"] = self.metrics

        # Neural logic overrides
        if self.is_neural:
            if max_count > 1:
                if self.use_fallback:
                    self.fallback_count += 1
                    action = heuristic_agent.agent(obs)
                else:
                    # Naive top 1 for no fallback (might crash)
                    action = [0]
            else:
                try:
                    action = self.agent_func(obs)
                except Exception as e:
                    print(f"Exception inside AgentWrapper for {self.agent_func}: {e}")
                    if self.use_fallback:
                        self.fallback_count += 1
                        action = heuristic_agent.agent(obs)
                    else:
                        action = [0]
        else:
            action = self.agent_func(obs)
            
        dt = (time.perf_counter() - start_t) * 1000.0
        self.decision_times.append(dt)
        self.total_decisions += 1
        
        return action

def run_tournament(agent1_name, agent1_fallback, agent2_name, num_games):
    def get_agent(name, fallback):
        if name == "random":
            return AgentWrapper("random", random_agent.agent, random_agent.read_deck_csv())
        elif name == "heuristic":
            return AgentWrapper("heuristic", heuristic_agent.agent, heuristic_agent.my_deck)
        elif name == "neural_v0":
            return AgentWrapper("neural_v0", neural_v0_agent_func, neural_deck, is_neural=True, use_fallback=fallback)
        elif name == "neural_v1":
            from src.nn.neural_agent_v1 import agent as neural_v1_agent_func
            return AgentWrapper("neural_v1", neural_v1_agent_func, neural_deck, is_neural=True, use_fallback=fallback)
        elif name == "neural_v1_belief":
            from src.nn.neural_agent_v1_belief import agent as neural_v1_belief_agent_func
            return AgentWrapper("neural_v1_belief", neural_v1_belief_agent_func, neural_deck, is_neural=True, use_fallback=fallback)
        elif name == "one_ply_search":
            from src.nn.one_ply_search_agent import agent as one_ply_search_agent_func
            return AgentWrapper("one_ply_search", one_ply_search_agent_func, neural_deck, is_neural=True, use_fallback=fallback)
        elif name == "neural_v2":
            from src.nn.neural_agent_v2 import agent as neural_v2_agent_func
            return AgentWrapper("neural_v2", neural_v2_agent_func, neural_deck, is_neural=True, use_fallback=fallback)
        elif name == "neural_v2_search_strict":
            from src.nn import neural_v2_search_agent
            neural_v2_search_agent.STRICT_SEARCH = True
            return AgentWrapper("neural_v2_search", neural_v2_search_agent.agent, neural_deck, is_neural=True, use_fallback=fallback)
        elif name == "neural_v2_search":
            from src.nn import neural_v2_search_agent
            neural_v2_search_agent.STRICT_SEARCH = False
            return AgentWrapper("neural_v2_search", neural_v2_search_agent.agent, neural_deck, is_neural=True, use_fallback=fallback)
        else:
            raise ValueError(f"Unknown agent: {name}")

    wins = 0
    losses = 0
    crashes = 0
    timeouts = 0
    game_lengths = []
    
    print(f"\nStarting Tournament: {agent1_name} (Fallback={agent1_fallback}) vs {agent2_name}")
    
    p1 = get_agent(agent1_name, agent1_fallback)
    p2 = get_agent(agent2_name, True)
    
    for i in range(num_games):
        try:
            obs_dict, _ = battle_start(p1.deck, p2.deck)
        except Exception as e:
            crashes += 1
            print(f"Game {i} failed to start: {e}")
            continue
            
        step_count = 0
        crashed = False
        
        while True:
            if obs_dict.get("current") and obs_dict["current"]["result"] != -1:
                # 0 if p1 wins, 1 if p2 wins
                res = obs_dict["current"]["result"]
                if res == 0:
                    wins += 1
                else:
                    losses += 1
                game_lengths.append(step_count)
                break
                
            try:
                if obs_dict.get("current")["yourIndex"] == 0:
                    action = p1.select_action(obs_dict)
                else:
                    action = p2.select_action(obs_dict)
                    
                obs_dict = battle_select(action)
            except IndexError:
                # Illegal action
                if obs_dict.get("current")["yourIndex"] == 0:
                    p1.illegal_actions += 1
                crashed = True
                crashes += 1
                break
            except Exception as e:
                crashed = True
                crashes += 1
                break
                
            step_count += 1
            if step_count > 1000:
                timeouts += 1
                break

    # Aggregate stats for P1
    if len(p1.decision_times) > 0:
        avg_dt = sum(p1.decision_times) / len(p1.decision_times)
        max_dt = max(p1.decision_times)
        if len(p1.decision_times) >= 20:
            p95_dt = statistics.quantiles(p1.decision_times, n=20)[18]
        else:
            p95_dt = max_dt
    else:
        avg_dt, max_dt, p95_dt = 0, 0, 0
        
    win_rate = (wins / num_games) * 100 if num_games > 0 else 0
    fallback_rate = (p1.fallback_count / p1.total_decisions) * 100 if p1.total_decisions > 0 else 0
    multi_rate = (p1.multi_select_count / p1.total_decisions) * 100 if p1.total_decisions > 0 else 0
    avg_steps = sum(game_lengths) / len(game_lengths) if game_lengths else 0
    median_steps = statistics.median(game_lengths) if game_lengths else 0
    
    import math
    p = wins / num_games if num_games > 0 else 0
    ci = 1.96 * math.sqrt((p * (1 - p)) / num_games) * 100 if num_games > 0 else 0
    
    print("-" * 50)
    print(f"Tournament Results: {agent1_name} vs {agent2_name}")
    print(f"Games Played:       {num_games}")
    print(f"Wins:               {wins}")
    print(f"Losses:             {losses}")
    print(f"Win Rate:           {win_rate:.1f}% ± {ci:.1f}% (95% CI)")
    print(f"Crashes:            {crashes}")
    print(f"Timeouts:           {timeouts}")
    print(f"Illegal Actions:    {p1.illegal_actions}")
    print(f"Average Steps:      {avg_steps:.1f}")
    print(f"Median Steps:       {median_steps}")
    print(f"Avg Decision Time:  {avg_dt:.2f} ms")
    print(f"p95 Decision Time:  {p95_dt:.2f} ms")
    print(f"Max Decision Time:  {max_dt:.2f} ms")
    print(f"Fallback Count:     {p1.fallback_count}")
    print(f"Fallback Rate:      {fallback_rate:.1f}%")
    print(f"Neural-only Rate:   {100 - fallback_rate:.1f}%")
    print(f"MultiSelect Count:  {p1.multi_select_count}")
    print(f"MultiSelect Rate:   {multi_rate:.1f}%")
    print("-" * 50)

    if "neural_v2_search" in p1.name:
        p1.metrics.print_report()
        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, default="neural_v1", choices=["random", "heuristic", "neural_v0", "neural_v1", "neural_v1_belief", "one_ply_search", "neural_v2", "neural_v2_search", "neural_v2_search_strict"])
    parser.add_argument("--fallback", type=str, default="none", choices=["heuristic", "none"])
    parser.add_argument("--opponent", type=str, default="random", choices=["random", "heuristic"])
    parser.add_argument("--games", type=int, default=100)
    
    args = parser.parse_args()
    use_fallback = (args.fallback == "heuristic")
    run_tournament(args.agent, use_fallback, args.opponent, args.games)
