import json
import torch
from torch.utils.data import Dataset, DataLoader

class ImitationDataset(Dataset):
    def __init__(self, jsonl_paths=None, max_cards=100, max_options=50, split="all", max_lines=200000):
        self.data = []
        
        if jsonl_paths is not None:
            if isinstance(jsonl_paths, str):
                jsonl_paths = [jsonl_paths]
                
            count = 0
            line_idx = 0
            for path in jsonl_paths:
                if count >= max_lines:
                    break
                with open(path, 'r') as f:
                    for line in f:
                        if count >= max_lines:
                            break
                            
                        if not line.strip(): continue
                        line_idx += 1
                        step_data = json.loads(line)
                        
                        if step_data["obs"].get("select") is None:
                            continue
                            
                        game_id = step_data.get("game_id")
                        if game_id is None:
                            # Fallback to deterministic pseudo-random split based on line_idx
                            game_id = line_idx // 100 # roughly group steps by game
                        split_bucket = game_id % 10
                        
                        if split == "train" and split_bucket < 8:
                            self.data.append(step_data)
                            count += 1
                        elif split == "val" and split_bucket == 8:
                            self.data.append(step_data)
                            count += 1
                        elif split == "test" and split_bucket == 9:
                            self.data.append(step_data)
                            count += 1
                        elif split == "all":
                            self.data.append(step_data)
                            count += 1
                        
        self.max_cards = max_cards
        self.max_options = max_options

    def __len__(self):
        return len(self.data)

    def _parse_global_features(self, current, step_data=None):
        if current is None:
            return torch.zeros(14 + 5 + 5) # 14 original + 5 deck_id + 5 matchup_id
        
        # turn, phase, prizes_left, etc.
        g = [
            current.get("turn", 0),
            current.get("turnActionCount", 0),
            len(current["players"][0].get("prize", [])),
            len(current["players"][1].get("prize", [])),
            current["players"][0].get("handCount", 0),
            current["players"][1].get("handCount", 0),
            current["players"][0].get("deckCount", 0),
            current["players"][1].get("deckCount", 0),
            len(current["players"][0].get("discard", [])),
            len(current["players"][1].get("discard", [])),
            float(current.get("energyAttached", False)),
            float(current.get("supporterPlayed", False)),
            float(current.get("stadiumPlayed", False)),
            float(current.get("retreated", False))
        ]
        
        # Add deck_id one-hot (0-4)
        deck_id = step_data.get("deck_id", 0) if step_data else 0
        matchup_id = step_data.get("matchup_id", 0) if step_data else 0
        
        for i in range(5):
            g.append(1.0 if i == deck_id else 0.0)
            
        for i in range(5):
            g.append(1.0 if i == matchup_id else 0.0)
            
        return torch.tensor(g, dtype=torch.float32)

    def _parse_cards(self, current, your_index):
        # We will parse active, bench, hand, discard for player 0 and player 1
        card_tokens = []
        numeric_features = []
        zone_masks = []
        
        # Zone definitions for aggregation: MyHand(0), MyActive(1), MyBench(2), MyDiscard(3), OpActive(4), OpBench(5), OpDiscard(6)
        
        if current is None:
            # Dummy return if no state
            return (
                torch.zeros((self.max_cards, 4), dtype=torch.long),
                torch.zeros((self.max_cards, 6), dtype=torch.float32),
                torch.zeros((self.max_cards, 7), dtype=torch.float32)
            )

        def add_pokemon(pkmn, owner, zone_agg, zone_raw, slot_id):
            if pkmn is None: return
            c_id = pkmn.get("id", 0)
            hp = pkmn.get("hp", 0)
            max_hp = pkmn.get("maxHp", 100)
            damage = max_hp - hp
            energies = len(pkmn.get("energies", []))
            is_active = 1 if zone_agg in [1, 4] else 0
            
            card_tokens.append([c_id, zone_raw, owner, slot_id])
            numeric_features.append([damage, hp, energies, 0, is_active, 0])
            
            zm = [0]*7
            zm[zone_agg] = 1
            zone_masks.append(zm)
            
        def add_card(card, owner, zone_agg, zone_raw, slot_id):
            if card is None: return
            c_id = card.get("id", 0)
            
            card_tokens.append([c_id, zone_raw, owner, slot_id])
            numeric_features.append([0, 0, 0, 0, 0, 0])
            
            zm = [0]*7
            zm[zone_agg] = 1
            zone_masks.append(zm)

        for player_idx in [0, 1]:
            p_state = current["players"][player_idx]
            owner = 0 if player_idx == your_index else 1
            
            # Active (Zone agg: 1 or 4)
            agg = 1 if owner == 0 else 4
            for s, pkmn in enumerate(p_state.get("active", [])):
                add_pokemon(pkmn, owner, agg, 4, s) # 4 is AreaType.ACTIVE
                
            # Bench (Zone agg: 2 or 5)
            agg = 2 if owner == 0 else 5
            for s, pkmn in enumerate(p_state.get("bench", [])):
                add_pokemon(pkmn, owner, agg, 5, s) # 5 is AreaType.BENCH
                
            # Discard (Zone agg: 3 or 6)
            agg = 3 if owner == 0 else 6
            for s, card in enumerate(p_state.get("discard", [])):
                add_card(card, owner, agg, 3, s) # 3 is AreaType.DISCARD
                
            # Hand (Zone agg: 0) - Only parse our hand
            if owner == 0 and p_state.get("hand"):
                for s, card in enumerate(p_state["hand"]):
                    add_card(card, owner, 0, 2, s) # 2 is AreaType.HAND

        # Pad to max_cards
        n = len(card_tokens)
        if n > self.max_cards:
            card_tokens = card_tokens[:self.max_cards]
            numeric_features = numeric_features[:self.max_cards]
            zone_masks = zone_masks[:self.max_cards]
        else:
            pad = self.max_cards - n
            card_tokens.extend([[0,0,0,0]] * pad)
            numeric_features.extend([[0,0,0,0,0,0]] * pad)
            zone_masks.extend([[0]*7] * pad)
            
        return (
            torch.tensor(card_tokens, dtype=torch.long),
            torch.tensor(numeric_features, dtype=torch.float32),
            torch.tensor(zone_masks, dtype=torch.float32)
        )

    def _parse_actions(self, options):
        # We need action_type, source_card_id, target_card_id, target_zone, target_slot, energy_type
        # boolean_flags, numeric_features, action_mask
        
        act_type = []
        src_c_id = []
        tgt_c_id = []
        tgt_zone = []
        tgt_slot = []
        e_type = []
        bool_flags = []
        num_feats = []
        act_mask = []
        
        for opt in options:
            act_type.append(opt.get("type", 0))
            
            # A lot of fields are optional or contextual in cg.api Option
            # We'll just map raw OptionType fields as cleanly as we can for plumbing.
            # E.g. "cardId"
            src_c_id.append(opt.get("cardId") or 0)
            
            # "area" -> target_zone
            tgt_zone.append(opt.get("area") or opt.get("inPlayArea") or 0)
            
            # "index" -> target_slot
            tgt_slot.append(opt.get("index") or opt.get("inPlayIndex") or 0)
            
            e_type.append(opt.get("energyIndex", 0) if opt.get("energyIndex") is not None else 0)
            
            # Target card ID might be inferable if it's EVOLVE or ATTACH, but we'll leave it 0 if not provided
            tgt_c_id.append(0)
            
            b1 = 1 if opt.get("type") == 14 else 0 # does_end_turn (END)
            b2 = 1 if opt.get("type") == 13 else 0 # does_attack (ATTACK)
            b3 = 1 if opt.get("type") == 8 else 0  # does_attach (ATTACH)
            b4 = 1 if opt.get("type") == 12 else 0 # does_retreat (RETREAT)
            b5 = 1 if opt.get("type") == 7 else 0  # does_play (PLAY)
            
            bool_flags.append([b1, b2, b3, b4, b5])
            
            n1 = opt.get("attackId") or 0
            n2 = opt.get("number") or opt.get("count") or 0
            num_feats.append([n1, n2])
            
            act_mask.append(1)
            
        # Pad to max_options
        n = len(options)
        if n > self.max_options:
            print(f"Warning: truncation! Num options: {n} > {self.max_options}")
            act_type = act_type[:self.max_options]
            src_c_id = src_c_id[:self.max_options]
            tgt_c_id = tgt_c_id[:self.max_options]
            tgt_zone = tgt_zone[:self.max_options]
            tgt_slot = tgt_slot[:self.max_options]
            e_type = e_type[:self.max_options]
            bool_flags = bool_flags[:self.max_options]
            num_feats = num_feats[:self.max_options]
            act_mask = act_mask[:self.max_options]
        else:
            pad = self.max_options - n
            act_type.extend([0] * pad)
            src_c_id.extend([0] * pad)
            tgt_c_id.extend([0] * pad)
            tgt_zone.extend([0] * pad)
            tgt_slot.extend([0] * pad)
            e_type.extend([0] * pad)
            bool_flags.extend([[0,0,0,0,0]] * pad)
            num_feats.extend([[0,0]] * pad)
            act_mask.extend([0] * pad)
            
        return {
            "action_type": torch.tensor(act_type, dtype=torch.long),
            "source_card_id": torch.tensor(src_c_id, dtype=torch.long),
            "target_card_id": torch.tensor(tgt_c_id, dtype=torch.long),
            "target_zone": torch.tensor(tgt_zone, dtype=torch.long),
            "target_slot": torch.tensor(tgt_slot, dtype=torch.long),
            "energy_type": torch.tensor(e_type, dtype=torch.long),
            "boolean_flags": torch.tensor(bool_flags, dtype=torch.float32),
            "numeric_features": torch.tensor(num_feats, dtype=torch.float32),
            "action_mask": torch.tensor(act_mask, dtype=torch.float32)
        }

    def __getitem__(self, idx):
        step_data = self.data[idx]
        current = step_data["obs"].get("current")
        obs = step_data["obs"]
        action = step_data.get("action_chosen", step_data.get("action"))
        
        current = obs.get("current")
        your_index = current.get("yourIndex", 0) if current else 0
        
        # Parse Deck ID
        deck_name = step_data.get("deck", "dragapult")
        deck_map = {"dragapult": 0, "abomasnow": 1, "lucario": 2, "iono": 3}
        deck_id = deck_map.get(deck_name, 0)
        
        global_features = self._parse_global_features(current, step_data)
        card_tokens, numeric_features, zone_masks = self._parse_cards(current, your_index)
        
        options = obs["select"].get("option", [])
        parsed_actions = self._parse_actions(options)
        
        # Extract the target action (single prediction for now)
        action_list = action if isinstance(action, list) else [action]
        # Multi-select could return empty list. For imitation we default to [0] or ignore.
        target_idx = action_list[0] if len(action_list) > 0 else 0
        if target_idx >= self.max_options:
            target_idx = -1 # Ignored by CrossEntropyLoss
        
        return {
            "state": {
                "deck_id": torch.tensor(deck_id, dtype=torch.long),
                "card_tokens": card_tokens,
                "numeric_features": numeric_features,
                "global_features": global_features,
                "zone_masks": zone_masks
            },
            "actions": parsed_actions,
            "target": torch.tensor(target_idx, dtype=torch.long)
        }

def get_dataloader(jsonl_path, batch_size=32, shuffle=True):
    dataset = ImitationDataset(jsonl_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
