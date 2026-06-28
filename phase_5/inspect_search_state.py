import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from cg.game import battle_start
from cg.api import search_begin, search_step
import agents.dragapult_agent as dragapult_agent
from src.nn import neural_v3_search_agent

obs, _ = battle_start(dragapult_agent.my_deck, dragapult_agent.my_deck)
opp_hand, opp_prize, opp_deck, opp_active = neural_v3_search_agent.build_determinization(obs)

agent_obs = neural_v3_search_agent.to_observation_class(obs)

my_state = obs["current"]["players"][0]
my_deck_count = my_state.get("deckCount", 0) or 0
my_prize_count = len(my_state.get("prize") or [])

your_deck_pred = dragapult_agent.my_deck[:my_deck_count]
your_prize_pred = dragapult_agent.my_deck[:my_prize_count]

ss = search_begin(
    agent_observation=agent_obs,
    your_deck=your_deck_pred, 
    your_prize=your_prize_pred,
    opponent_deck=opp_deck,
    opponent_prize=opp_prize,
    opponent_hand=opp_hand,
    opponent_active=opp_active,
    manual_coin=0
)

# Print attributes of ss.observation
print("ss.observation keys/attrs:", dir(ss.observation))
if hasattr(ss.observation, 'select') and ss.observation.select:
    print("select:", dir(ss.observation.select))
    if hasattr(ss.observation.select, 'option'):
        print("options:", ss.observation.select.option)
