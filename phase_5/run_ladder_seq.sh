#!/bin/bash
echo "Running Random Validation..."
docker run --platform linux/amd64 --rm -v "$(pwd):/app" ptcg-runner-phase2 python /app/phase_5/src/eval/strict_tournament.py --agent neural_v2_search --fallback heuristic --opponent random --games 200 > ladder_random.log 2>&1
echo "Random complete."

for opp in dragapult abomasnow lucario iono; do
    echo "Running ladder against $opp..."
    docker run --platform linux/amd64 --rm -v "$(pwd):/app" ptcg-runner-phase2 python /app/phase_5/src/eval/ladder.py --games 100 --opponents $opp > ladder_${opp}.log 2>&1
    echo "$opp complete."
done
echo "All ladder matches complete."
