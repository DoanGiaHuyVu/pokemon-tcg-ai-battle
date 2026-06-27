# Phase 4: Hybrid Search Integration

This phase transitions the agent from a pure neural policy into a hybrid search framework, combining neural priors with a deterministic 1-ply tactical search.

## Key Accomplishments

- Integrated the C++ simulator API directly into the agent decision loop.
- Developed the `neural_v2_search_agent` which utilizes `search_begin` and `search_step` to preview game states.
- Implemented a tactical evaluator to score the resulting board states based on damage math and energy curves.
- Proven the hybrid neural-search concept can interface with the environment, revealing the necessity for rigorous determinization.

## Key Files & Functions

- **`src/nn/neural_v2_search_agent.py`**: The `agent(obs)` function intercepts the neural prior, explores the game tree via the C++ `search_begin` and `search_step` APIs, and returns the highest-scoring evaluated move.
- **`src/nn/tactical_evaluator.py`**: The `evaluate(obs)` function dynamically scores a specific future board state utilizing structural rules like damage tracking and energy curves.
- **`src/nn/one_ply_search_agent.py`**: A foundational script used to prototype and prove that the C++ bindings could be called directly from Python before integrating it with neural priors.
