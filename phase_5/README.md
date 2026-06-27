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
