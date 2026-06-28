# Phase 5: Search Validation & Opponent Ladder

This phase rigorously stabilizes the search system by eliminating silent fallback bugs, fixing determinization logic, and preparing the agent for the opponent ladder.

## Key Focus Areas

- Implementing a global `metrics.py` registry to track actual search attempts, successes, and exceptions.
- Fixing exact-count determinization logic (hand, deck, active pokemon) to ensure the simulator accepts mocked opponent hidden states.
- Re-evaluating the true search-driven win rate using `strict_tournament.py` without silent fallbacks.
- Evaluating the final search agent against a diverse opponent ladder.

## Key Files & Functions

- **`src/eval/metrics.py`**: The `MetricsRegistry` class serves as a global tracking instance injected into agents to faithfully record hidden fallback triggers and search loop exceptions.
- **`src/eval/strict_tournament.py`**: Uses an `AgentWrapper` that intercepts and enforces `STRICT_SEARCH` validation, crashing the game intentionally when search assumptions fail.
- **`src/search_api_probe.py`**: The `run_probe` script is specifically tailored to hook into `cg.api` to dump memory structures, types, and attributes (such as `SearchState` mapping) directly from the Pybind11 simulator layer.

## Search Stabilization Results

After identifying a critical "silent fallback illusion" where the agent appeared to have a 100% win rate due to failing searches silently falling back to the heuristic agent, the determinization logic and API signatures were strictly fixed.

**Post-Fix Integrity Metrics:**
- **Search Begin Success Rate:** 99.0%
- **Search Step Success Rate:** 100.0%
- **Action Evaluation Coverage:** 100.0%
- **Fallback Rate:** 0.9%

**The Exposed Truth:**
With the silent fallbacks eliminated, the true win rate of the hybrid search agent vs. Random was revealed to be **22%**. This proves that the search plumbing is now robust, but the underlying tactical heuristic evaluator is currently too weak (only scoring prize leads) and needs significant strengthening in future work.

## Phase 6: Strengthen Tactical Evaluator

To address the 22% win rate, we completely overhauled the tactical scoring mechanism to use an **action delta-based approach** (`score_action = score_state(next_state) - score_state(current_state)`). This ensures the agent is rewarded for actions that *improve* the board, rather than rewarding passive play when already in a winning state.

### Key Changes
1. **Face-Down Active Prediction:** The agent now dynamically identifies Basic Pokémon from the card metadata to predict the opponent's face-down active slot, successfully pushing the `search_begin` success rate from 99.0% to **100.0%**.
2. **Delta-Based Evaluator (`tactical_evaluator.py`):** Rewrote the evaluator to score:
   - **KO Proximity:** Rewards damage relative to Max HP, with bonuses for actual KOs and placing opponents in next-attack KO range.
   - **Energy Tempo:** Rewards useful energy attachments (+20) and penalizes opponent energy.
   - **Missed Attack / Missed KO Guards:** Severely punishes the agent (-300 / -700) for passing the turn when an attack or KO was available.
   - **Board & Hand Quality:** Modest conservative bonuses for filling the bench and drawing cards.
3. **Weight Normalization:** Adjusted the Neural Prior weight (50x) vs Tactical Delta (1x) in `neural_v2_search_agent.py` so the search engine's discoveries aren't drowned out by the neural network's baseline habits.

### Tested Strategies & Results

We evaluated the `neural_v2_search` hybrid agent using `strict_tournament.py` across two 100-game match-ups:

**1. vs Random (The Primary Baseline):**
- **Win Rate:** **41.0% ± 9.6%** (up from 22%).
- **Integrity:** 100.0% `search_begin`, 100.0% `search_step`, 0% fallback rate.
- **Analysis:** The win rate nearly doubled. The agent is measurably smarter, aggressively seeking KOs, attaching useful energy, and refusing to pass the turn when attacks are available.

**2. vs Dragapult Heuristic (Diagnostic):**
- **Win Rate:** **2.0% ± 2.7%**.
- **Analysis:** This confirms a massive knowledge gap. The heuristic bot possesses deck-specific tactical combos (e.g., Phantom Dive spread math, Rare Candy evolution lines, Boss's Orders targeting) that our general-purpose evaluator does not yet understand. The next milestone is pushing the Random win rate to >70% via deeper MCTS or learned value functions before tackling the expert heuristic bots.
