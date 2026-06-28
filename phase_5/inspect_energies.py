import json

with open('data/dataset_v3_expert.jsonl') as f:
    for line in f:
        data = json.loads(line)
        for p in data['obs']['current']['players']:
            for pkmn in (p.get('active', []) + p.get('bench', [])):
                if not pkmn: continue
                if 'energies' in pkmn and len(pkmn['energies']) > 0:
                    print("energies:", pkmn['energies'])
                    print("energyCards:", pkmn.get('energyCards'))
                    exit(0)
