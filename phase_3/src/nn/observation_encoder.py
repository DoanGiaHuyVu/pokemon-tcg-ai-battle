import torch
import torch.nn as nn
import torch.nn.functional as F

class ObservationEncoder(nn.Module):
    def __init__(
        self,
        d_model=128,
        num_cards=1500,  # Max card IDs
        num_zones=15,
        num_owners=3,    # 0=Player, 1=Opponent, 2=Neutral
        num_slots=60,    # Max slots in any zone
        num_numeric_features=6, # damage, hp, energies, conditions, is_active, evolution
        num_global_features=14
    ):
        super().__init__()
        self.d_model = d_model
        
        # Embeddings
        self.card_emb = nn.Embedding(num_cards, d_model)
        self.zone_emb = nn.Embedding(num_zones, 16)
        self.owner_emb = nn.Embedding(num_owners, 8)
        self.slot_emb = nn.Embedding(num_slots, 8)
        
        # Numeric Projection
        self.numeric_proj = nn.Linear(num_numeric_features, 16)
        
        # Token Projection (card + zone + owner + slot + numeric) -> d_model
        token_dim = d_model + 16 + 8 + 8 + 16
        self.token_proj = nn.Sequential(
            nn.Linear(token_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        # Global Features Projection
        self.global_proj = nn.Sequential(
            nn.Linear(num_global_features, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        
        # Final State MLP (concatenated pooled zones + global) -> d_model
        # We will use 7 aggregated zones: MyHand, MyActive, MyBench, MyDiscard, OpActive, OpBench, OpDiscard.
        self.num_aggregated_zones = 7
        self.state_mlp = nn.Sequential(
            nn.Linear(self.num_aggregated_zones * d_model + 64, 256),
            nn.ReLU(),
            nn.Linear(256, d_model)
        )

    def forward(self, card_tokens, numeric_features, global_features, zone_masks):
        # card_tokens: (batch_size, max_cards, 4) -> [card_id, zone_id, owner_id, slot_id]
        # numeric_features: (batch_size, max_cards, num_numeric_features)
        # global_features: (batch_size, num_global_features)
        # zone_masks: (batch_size, max_cards, num_aggregated_zones) - one-hot mask for which zone a card belongs to
        
        b, n, _ = card_tokens.shape
        
        c_id = self.card_emb(card_tokens[:, :, 0])
        z_id = self.zone_emb(card_tokens[:, :, 1])
        o_id = self.owner_emb(card_tokens[:, :, 2])
        s_id = self.slot_emb(card_tokens[:, :, 3])
        
        n_feat = self.numeric_proj(numeric_features)
        
        # Concat all token features
        raw_tokens = torch.cat([c_id, z_id, o_id, s_id, n_feat], dim=-1)
        
        # Project tokens to d_model
        tokens = self.token_proj(raw_tokens) # (b, n, d_model)
        
        # Pool by zone
        # zone_masks shape: (b, n, num_aggregated_zones)
        zone_counts = zone_masks.sum(dim=1, keepdim=True).clamp(min=1e-9) # (b, 1, num_aggregated_zones)
        zone_counts = zone_counts.transpose(1, 2) # (b, num_aggregated_zones, 1)
        
        # (b, num_aggregated_zones, n) @ (b, n, d_model) -> (b, num_aggregated_zones, d_model)
        zone_sum = torch.bmm(zone_masks.transpose(1, 2), tokens)
        
        zone_mean = zone_sum / zone_counts # Mean pooling per zone
        zone_flat = zone_mean.view(b, -1) # (b, num_aggregated_zones * d_model)
        
        # Global features
        g_feat = self.global_proj(global_features) # (b, 64)
        
        # Combine
        combined = torch.cat([zone_flat, g_feat], dim=-1)
        state_vector = self.state_mlp(combined) # (b, d_model)
        
        return state_vector
