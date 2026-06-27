# Phase 3: Self-Play and Evaluation

This phase expands upon the Phase 2 baseline by focusing on rigorous evaluation and data expansion.

## Key Accomplishments

- Established a tournament evaluator (`eval_tournament.py`) to systematically measure agent win rates and decision times.
- Iteratively improved the neural network's capabilities through self-play data.
- Refined the evaluation framework to properly distinguish between pure neural decisions and heuristic fallback decisions.

## Key Files & Functions

- **`src/eval/eval_tournament.py`**: The `run_tournament` function sets up automated match loops to scientifically record win rates and decision times between diverse agents.
- **`src/nn/train_imitation.py`**: The `train_model` function handles supervised learning loops, feeding simulated self-play datasets into the neural baseline.
