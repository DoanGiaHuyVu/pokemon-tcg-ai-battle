import torch
import torch.nn as nn

class ActionEncoder(nn.Module):
    def __init__(
        self,
        d_model=128,
        num_action_types=20, # Number of OptionType enums
        num_cards=1500,
        num_zones=15,
        num_slots=60,
        num_energy_types=15,
        num_boolean_flags=5
    ):
        super().__init__()
        
        self.d_model = d_model
        
        self.action_type_emb = nn.Embedding(num_action_types, 16)
        
        # Source/Target embeddings (reuse sizes)
        self.card_emb = nn.Embedding(num_cards, 32)
        self.zone_emb = nn.Embedding(num_zones, 16)
        self.slot_emb = nn.Embedding(num_slots, 16)
        self.energy_emb = nn.Embedding(num_energy_types, 8)
        
        # Booleans
        self.bool_proj = nn.Linear(num_boolean_flags, 16)
        
        # Other numeric like attack_index, raw_option_index
        self.numeric_proj = nn.Linear(2, 16)
        
        # Total dimension:
        # action_type (16) + source_card (32) + target_card (32) + target_zone (16) + target_slot (16) 
        # + energy (8) + bools (16) + numeric (16) = 152
        
        self.mlp = nn.Sequential(
            nn.Linear(152, 256),
            nn.ReLU(),
            nn.Linear(256, d_model)
        )
        
    def forward(self, action_features):
        # action_features is a dict of Tensors, each shape (batch_size, num_options)
        # For pointer networks, we usually process all options in parallel.
        
        b, num_opt = action_features["action_type"].shape
        
        t_emb = self.action_type_emb(action_features["action_type"])
        src_c_emb = self.card_emb(action_features["source_card_id"])
        tgt_c_emb = self.card_emb(action_features["target_card_id"])
        tgt_z_emb = self.zone_emb(action_features["target_zone"])
        tgt_s_emb = self.slot_emb(action_features["target_slot"])
        e_emb = self.energy_emb(action_features["energy_type"])
        
        b_feat = self.bool_proj(action_features["boolean_flags"])
        n_feat = self.numeric_proj(action_features["numeric_features"])
        
        # Concatenate along feature dimension
        combined = torch.cat([
            t_emb, src_c_emb, tgt_c_emb, tgt_z_emb, tgt_s_emb, e_emb, b_feat, n_feat
        ], dim=-1)
        
        action_vectors = self.mlp(combined) # (b, num_options, d_model)
        return action_vectors
