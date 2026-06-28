# Pokémon TCG AI Battle Challenge Strategy

This project contains an AI agent designed for the Pokémon TCG AI Battle Challenge Strategy competition on Kaggle. 

## Project Structure

- `phase_2`: Neural network behavior cloning baseline implementation.
- `phase_3`: Self-play expansion, reinforcement learning improvements, and evaluation infrastructure.
- `phase_4`: Initial integration of a 1-ply search system combining the neural prior with a tactical evaluation heuristic.
- `phase_5`: Robust search validation and deterministic simulator bug fixing. Successfully stabilized the search engine (99% API success, 0.9% fallback), exposing the true baseline search win rate of 22% vs. Random.
- `phase_6`: (Included in `phase_5/` codebase). Strengthened tactical evaluator using an action delta-based approach (KO proximity, missed-attack guards). Improved win rate to 41% vs. Random and achieved 100% search API success.
- `phase_6.5`: Upgraded to a Turn-Level Beam Search (width 3, depth 8) and `turns_to_attack` simulated damage evaluation. Achieved **100% win rate vs Random** and 6% against the expert Heuristic bot.
- `phase_7`: Conducted loss analysis on the heuristic bottleneck, revealing a major blind spot in the board evaluator (ignoring benched Pokémon). Rewrote the tactical evaluator for board-wide scoring and achieved a **30% win rate vs the Expert Dragapult heuristic**.
This project uses a hybrid approach, combining a neural network policy for broad strategy with a tactical search system for precise decision-making in the simulator.

## Setup & Execution

### Prerequisites
- Docker (with `linux/amd64` platform support)
- The pre-built `ptcg-runner-phase2` Docker image containing the Pokémon TCG C++ simulator engine and PyTorch dependencies.

### Running Agents & Tournaments
Because the underlying C++ simulator bindings are compiled for the `ptcg-runner-phase2` container environment, all scripts must be executed inside the container by mounting the current directory.

**Example: Running the Phase 5 Strict Tournament Evaluation:**
```bash
docker run --platform linux/amd64 --rm -v "$(pwd):/app" ptcg-runner-phase2 python /app/phase_5/src/eval/strict_tournament.py --agent neural_v2_search_strict --fallback heuristic --opponent random --games 100
```

### Available Testing Modes & Agents

You can configure matches between any combination of agents using the `--agent` and `--opponent` flags in the tournament scripts (`eval_tournament.py` and `strict_tournament.py`).

**Available Agents:**
- `random`: Makes completely random legal moves.
- `dragapult`, `abomasnow`, `lucario`, `iono`: Hand-crafted deterministic heuristic agents specialized for specific starter decks.
- `neural_v2`: Pure neural network policy agent (Behavior Cloning).
- `neural_v2_search`: Hybrid agent using Neural prior + 1-Ply Tactical Search (Defaults to heuristic fallback on failure).
- `neural_v2_search_strict`: Hybrid search agent strictly enforcing search integrity (Crashes instead of falling back on unhandled API errors).

**Key Tournament Flags:**
- `--games <N>`: Set the number of matches to play.
- `--fallback <agent_name>`: Specify a backup agent (e.g., `heuristic`) to take over if the primary agent throws an exception.
- `--debug`: Enable verbose step-by-step console logging of the match.

**Example: Running the API Probe Diagnostics:**
```bash
docker run --platform linux/amd64 --rm -v "$(pwd):/app" ptcg-runner-phase2 python /app/phase_5/src/search_api_probe.py
```

### Important Notes
- Always mount your local repository to `/app` using `-v "$(pwd):/app"`.
- Use the `--platform linux/amd64` flag if you are on an Apple Silicon (M1/M2/M3) Mac to ensure compatibility with the pre-compiled simulator binaries.

## Important Files to Watch

When navigating or modifying this project, pay special attention to the following core files, as they form the backbone of the agent and evaluation loop:

1. **`phase_5/src/nn/neural_v2_search_agent.py`**: The crown jewel of the hybrid search architecture. This file handles environment determinization, interacts directly with the `cg.api` C++ simulator bindings, and integrates the PyTorch neural network.
2. **`phase_5/src/eval/strict_tournament.py`**: The primary evaluation harness. It orchestrates matches between agents, enforces strict search policies to prevent silent fallbacks, and outputs comprehensive win-rate statistics.
3. **`phase_5/src/eval/metrics.py`**: The global telemetry registry. This tracks all search successes, API exceptions, and heuristic fallbacks to ensure the integrity of the agent's decision-making process.
4. **`phase_5/src/nn/tactical_evaluator.py`**: The deterministic heuristic scorer. It evaluates and grades simulated future board states based on criteria like damage mapping, energy curves, and prize counts.
5. **`cg/api.py` & `cg/game.py`**: The Python wrappers over the C++ Pybind11 simulator. Understanding the structures here (e.g., `Observation`, `SearchState`, `State`) is crucial for interfacing with the Pokémon TCG engine.

## Data & Models Directory

- **`data/`**: Contains the raw and pre-processed JSON datasets of simulator self-play matches used for behavior cloning.
- **`models/`**: Stores the compiled `.pt` PyTorch model weights generated at the end of each training phase.
- **`EN_Card_Data.csv` & `JP_Card_Data.csv`**: Reference files containing all Pokémon TCG card metadata (HP, attacks, types, retreat costs) provided by the Kaggle competition.

## Core Dependencies

This project relies on the following core technologies, bundled within the `ptcg-runner-phase2` Docker container:
- **PyTorch**: Used for building and executing the behavior cloning neural network.
- **Pybind11**: Acts as the bridge between the high-performance C++ simulator engine and the Python agent logic.
- **Python 3.10+**: Standard execution environment for the scripts.


