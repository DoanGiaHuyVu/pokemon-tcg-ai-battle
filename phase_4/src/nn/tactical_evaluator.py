import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from cg.api import State

def score_state(state: State, your_index: int) -> float:
    """
    Heuristically scores a game state from the perspective of `your_index`.
    """
    # Result check
    if state.result != -1:
        if state.result == your_index:
            return 10000.0
        else:
            return -10000.0

    my_state = state.players[your_index]
    opp_state = state.players[1 - your_index]

    score = 0.0

    # 1. Prize Lead (+300 per prize)
    my_prizes_taken = 6 - len(my_state.prize)
    opp_prizes_taken = 6 - len(opp_state.prize)
    prize_lead = my_prizes_taken - opp_prizes_taken
    score += prize_lead * 300.0

    # 2. Board Setup
    # 2. Bench Development (Removed to let NN decide benching)
    # 5. Hand size (Removed to let NN decide draw cards)

    return score
