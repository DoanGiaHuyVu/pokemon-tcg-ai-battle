import json

with open('data/dataset_v3_expert.jsonl') as f:
    for line in f:
        data = json.loads(line)
        for p in data['obs']['current']['players']:
            for pkmn in (p.get('active', []) + p.get('bench', [])):
                if not pkmn: continue
                # Print the keys of a pokemon object!
                print(list(pkmn.keys()))
                if 'energies' in pkmn or 'energy' in pkmn or 'attached' in pkmn:
                    print(pkmn)
                    exit(0)
