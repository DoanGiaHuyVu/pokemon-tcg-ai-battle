#!/bin/bash
# Phase 7: Sharded Distillation Data Collector
# Runs data collection in small batches to prevent memory leaks

OUT_FILE="data/dataset_v3_expert.jsonl"
OUT_FILE_BEAM="data/dataset_v3_beam.jsonl"
GAMES_PER_BATCH=20

echo "Starting Expert Data Collection..."
for opp in dragapult abomasnow lucario iono; do
    echo "Collecting 1000 games for expert: $opp"
    for i in {1..50}; do
        docker run --platform linux/amd64 --rm -v "$(pwd):/app" ptcg-runner-phase2 python /app/phase_5/src/data/collect_multi_distillation.py --out $OUT_FILE --games $GAMES_PER_BATCH --p0 $opp --p1 random
    done
done

echo "Starting Beam Search Data Collection..."
# Collect 1000 games of beam search vs random to seed search distillation
for i in {1..50}; do
    echo "Beam search batch $i/50"
    docker run --platform linux/amd64 --rm -v "$(pwd):/app" ptcg-runner-phase2 python /app/phase_5/src/data/collect_multi_distillation.py --out $OUT_FILE_BEAM --games $GAMES_PER_BATCH --p0 beam_search --p1 random
done

echo "Data collection complete."
