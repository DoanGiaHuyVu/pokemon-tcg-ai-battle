import sys
import json
import torch
from torch.utils.data import Dataset, DataLoader
from src.nn.belief_tracker import BeliefTracker
from agents.dragapult_agent import my_deck as opponent_deck_list

class ImitationDataset(Dataset):
    def __init__(self, jsonl_path=None, max_cards=100, max_options=50, split="all"):
        self.data = []
        self.belief_tracker = BeliefTracker(opponent_deck_list)
        self.unique_cards = sorted(list(set(opponent_deck_list)))
        if jsonl_path is not None:
            with open(jsonl_path, 'r') as f:
                for line in f:
                    if not line.strip(): continue
                    step_data = json.loads(line)
                    
                    # We only want to train on steps where an action was actually taken
                    if step_data["obs"].get("select") is None:
                        continue
                        
                    game_id = step_data.get("game_id", 0)
                    
                    # 80/10/10 split
                    split_bucket = game_id % 10
                    
                    if split == "train" and split_bucket < 8:
                        self.data.append(step_data)
                    elif split == "val" and split_bucket == 8:
                        self.data.append(step_data)
                    elif split == "test" and split_bucket == 9:
                        self.data.append(step_data)
                    elif split == "all":
                        self.data.append(step_data)
                        
        self.max_cards = max_cards
        self.max_options = max_options

    def __len__(self):
        return len(self.data)

    def _parse_global_features(self, current):
        if current is None:
            return torch.zeros(14)
        
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
        
        # Add belief features
        self.belief_tracker.update({"current": current})
        belief_features = []
        for card_id in self.unique_cards:
            belief_features.append(self.belief_tracker.get_probability_in_hand(card_id))
            
        g.extend(belief_features)
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
        item = self.data[idx]
        obs = item["obs"]
        raw_action = item["action"]
        target_action = raw_action[0] if isinstance(raw_action, list) and len(raw_action) > 0 else (raw_action if isinstance(raw_action, int) else 0)
        value_target = item["game_result"] # 0 for P1, 1 for P2
        
        your_index = obs.get("current", {}).get("yourIndex", 0)
        
        # Value target: 1 if we won, -1 if we lost
        # (Assuming your_index == game_result means win)
        # However, Dragapult is always agent_0 in collect_imitation_data, so your_index is 0.
        # Winner = 0 means win.
        if value_target == your_index:
            v = 1.0
        elif value_target == 1 - your_index:
            v = -1.0
        else:
            v = 0.0 # Draw or unfinished
            
        # 1. Parse Global
        global_features = self._parse_global_features(obs.get("current"))
        
        # 2. Parse Cards
        card_tokens, numeric_features, zone_masks = self._parse_cards(obs.get("current"), your_index)
        
        state_inputs = {
            "card_tokens": card_tokens,
            "numeric_features": numeric_features,
            "global_features": global_features,
            "zone_masks": zone_masks
        }
        
        # 3. Parse Actions
        options = obs["select"]["option"] if obs.get("select") else []
        action_inputs = self._parse_actions(options)
        
        return state_inputs, action_inputs, torch.tensor(target_action, dtype=torch.long), torch.tensor(v, dtype=torch.float32)

def get_dataloader(jsonl_path, batch_size=32, shuffle=True):
    dataset = ImitationDataset(jsonl_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
