#!/usr/bin/env python3
"""
Phase 7: Opponent Ladder Script.
Runs the beam search agent against all available opponents and collects comprehensive stats.
"""
import argparse
import time
import json
import statistics
import os
import sys
import math

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from cg.game import battle_start, battle_select
import agents.random_agent as random_agent
import agents.dragapult_agent as dragapult_agent
import agents.mega_abomasnow_agent as abomasnow_agent
import agents.mega_lucario_agent as lucario_agent
import agents.iono_agent as iono_agent

from src.eval.metrics import MetricsRegistry

# Map of opponent name -> (agent_func, deck)
OPPONENT_REGISTRY = {
    "random": (random_agent.agent, random_agent.read_deck_csv()),
    "dragapult": (dragapult_agent.agent, dragapult_agent.my_deck),
    "abomasnow": (abomasnow_agent.agent, abomasnow_agent.my_deck),
    "lucario": (lucario_agent.agent, lucario_agent.my_deck),
    "iono": (iono_agent.agent, iono_agent.my_deck),
}


class AgentWrapper:
    def __init__(self, name, agent_func, deck, is_neural=False, use_fallback=True):
        self.name = name
        self.agent_func = agent_func
        self.deck = deck
        self.is_neural = is_neural
        self.use_fallback = use_fallback
        self.decision_times = []
        self.fallback_count = 0
        self.total_decisions = 0
        self.illegal_actions = 0
        self.metrics = MetricsRegistry()

    def select_action(self, obs):
        start_t = time.perf_counter()
        obs["metrics"] = self.metrics

        if self.is_neural:
            select_block = obs.get("select") or {}
            max_count = select_block.get("maxCount", 1)
            if max_count > 1:
                if self.use_fallback:
                    self.fallback_count += 1
                    action = dragapult_agent.agent(obs)
                else:
                    action = [0]
            else:
                try:
                    action = self.agent_func(obs)
                except RuntimeError:
                    raise
                except Exception as e:
                    if self.use_fallback:
                        self.fallback_count += 1
                        action = dragapult_agent.agent(obs)
                    else:
                        action = [0]
        else:
            action = self.agent_func(obs)

        dt = (time.perf_counter() - start_t) * 1000.0
        self.decision_times.append(dt)
        self.total_decisions += 1
        return action


def run_matchup(agent_name, opp_name, num_games, use_fallback=True):
    """Run a single matchup and return results dict."""
    from src.nn import neural_v2_search_agent
    from src.nn import neural_v3_search_agent

    agents = {
        "random": random_agent.agent,
        "dragapult": dragapult_agent.agent,
        "abomasnow": abomasnow_agent.agent,
        "lucario": lucario_agent.agent,
        "iono": iono_agent.agent,
        "beam_search": neural_v2_search_agent.agent,
        "beam_search_v3": neural_v3_search_agent.agent
    }
    
    neural_v2_search_agent.STRICT_SEARCH = False
    neural_v3_search_agent.STRICT_SEARCH = False
    neural_deck = dragapult_agent.my_deck

    p1 = AgentWrapper(agent_name, agents[agent_name], neural_deck,
                       is_neural=True, use_fallback=use_fallback)

    opp_func, opp_deck = OPPONENT_REGISTRY[opp_name]
    p2 = AgentWrapper(opp_name, opp_func, opp_deck)

    wins = 0
    losses = 0
    crashes = 0
    timeouts = 0
    game_lengths = []

    for i in range(num_games):
        try:
            obs_dict, _ = battle_start(p1.deck, p2.deck)
        except Exception as e:
            crashes += 1
            continue

        step_count = 0
        while True:
            if obs_dict.get("current") and obs_dict["current"]["result"] != -1:
                res = obs_dict["current"]["result"]
                if res == 0:
                    wins += 1
                else:
                    losses += 1
                game_lengths.append(step_count)
                break

            try:
                if obs_dict["current"]["yourIndex"] == 0:
                    action = p1.select_action(obs_dict)
                else:
                    action = p2.select_action(obs_dict)
                obs_dict = battle_select(action)
            except RuntimeError as e:
                crashes += 1
                break
            except IndexError:
                if obs_dict["current"]["yourIndex"] == 0:
                    p1.illegal_actions += 1
                crashes += 1
                break
            except Exception:
                crashes += 1
                break

            step_count += 1
            if step_count > 1000:
                timeouts += 1
                break

    # Stats
    if p1.decision_times:
        avg_dt = sum(p1.decision_times) / len(p1.decision_times)
        max_dt = max(p1.decision_times)
        p95_dt = statistics.quantiles(p1.decision_times, n=20)[18] if len(p1.decision_times) >= 20 else max_dt
    else:
        avg_dt = max_dt = p95_dt = 0

    win_rate = (wins / num_games) * 100 if num_games > 0 else 0
    avg_steps = sum(game_lengths) / len(game_lengths) if game_lengths else 0
    median_steps = statistics.median(game_lengths) if game_lengths else 0
    p = wins / num_games if num_games > 0 else 0
    ci = 1.96 * math.sqrt((p * (1 - p)) / num_games) * 100 if num_games > 0 else 0

    result = {
        "opponent": opp_name,
        "games": num_games,
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 1),
        "ci_95": round(ci, 1),
        "crashes": crashes,
        "timeouts": timeouts,
        "illegal_actions": p1.illegal_actions,
        "avg_steps": round(avg_steps, 1),
        "median_steps": round(median_steps, 1),
        "avg_dt_ms": round(avg_dt, 2),
        "p95_dt_ms": round(p95_dt, 2),
        "max_dt_ms": round(max_dt, 2),
        "fallback_rate": round((p1.fallback_count / p1.total_decisions) * 100, 1) if p1.total_decisions > 0 else 0,
        "search_begin_success": p1.metrics.search_begin_successes,
        "search_begin_fail": p1.metrics.search_begin_failures,
        "search_step_success": p1.metrics.search_step_successes,
        "search_step_fail": p1.metrics.search_step_failures,
    }
    return result


def print_result(r, agent_name="beam_search"):
    print(f"\n{'='*60}")
    print(f"  {agent_name} vs {r['opponent']}  ({r['games']} games)")
    print(f"{'='*60}")
    print(f"  Win Rate:        {r['win_rate']}% ± {r['ci_95']}%")
    print(f"  Wins/Losses:     {r['wins']}/{r['losses']}")
    print(f"  Crashes:         {r['crashes']}")
    print(f"  Timeouts:        {r['timeouts']}")
    print(f"  Avg Steps:       {r['avg_steps']}")
    print(f"  Avg Decision:    {r['avg_dt_ms']} ms")
    print(f"  p95 Decision:    {r['p95_dt_ms']} ms")
    print(f"  Max Decision:    {r['max_dt_ms']} ms")
    print(f"  Fallback Rate:   {r['fallback_rate']}%")
    print(f"  Search Begin:    {r['search_begin_success']} ok / {r['search_begin_fail']} fail")
    print(f"  Search Step:     {r['search_step_success']} ok / {r['search_step_fail']} fail")
    print(f"{'='*60}")


def print_ladder_summary(results):
    print(f"\n{'='*70}")
    print(f"  OPPONENT LADDER SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Opponent':<15} {'Win Rate':>10} {'CI 95%':>10} {'Games':>8} {'Crashes':>10}")
    print(f"  {'-'*55}")
    for r in results:
        print(f"  {r['opponent']:<15} {r['win_rate']:>8.1f}% {r['ci_95']:>8.1f}% {r['games']:>8} {r['crashes']:>10}")
    print(f"{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 7 Opponent Ladder")
    parser.add_argument("--agent", type=str, default="beam_search", help="Agent to test")
    parser.add_argument("--games", type=int, default=100, help="Games per matchup")
    parser.add_argument("--opponents", type=str, nargs="+",
                        default=["random", "dragapult", "abomasnow", "lucario", "iono"],
                        help="Opponents to test against")
    args = parser.parse_args()

    all_results = []
    for opp in args.opponents:
        if opp not in OPPONENT_REGISTRY:
            print(f"Unknown opponent: {opp}, skipping")
            continue
        r = run_matchup(args.agent, opp, args.games)
        print_result(r, agent_name=args.agent)
        all_results.append(r)

    print_ladder_summary(all_results)

    # Save JSON results
    out_path = os.path.join(os.path.dirname(__file__), "ladder_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")
