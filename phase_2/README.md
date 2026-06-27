# Phase 2: Neural Network Baseline

This phase focuses on training a baseline neural network using behavior cloning (Imitation Learning) on the provided dataset.

## Key Accomplishments

- Implemented an end-to-end tensor encoder for simulator observations.
- Created dynamic legal action scoring that successfully interfaces with the simulator.
- Achieved a stable model capable of playing matches from start to finish without crashing.
- Integrated a heuristic fallback mechanism to handle complex search-based actions.

## Key Files & Functions

- **`src/nn/model.py`**: Contains the core architecture for the behavior cloning neural network.
- **`src/nn/observation_encoder.py`**: Converts raw simulator JSON observations into multi-dimensional PyTorch tensors.
- **`src/nn/action_encoder.py`**: Translates complex, multi-variable Pokémon TCG actions into discrete option tokens.
- **`src/nn/neural_agent.py`**: The main interface function `agent(obs)` that consumes the neural model to predict and return the best action.
