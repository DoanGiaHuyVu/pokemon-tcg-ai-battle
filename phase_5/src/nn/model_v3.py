import torch
import torch.nn as nn
from src.nn.observation_encoder_v3 import ObservationEncoder
from src.nn.action_encoder import ActionEncoder

class PointerPolicyModel(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        self.d_model = d_model
        
        self.state_encoder = ObservationEncoder(d_model=d_model, global_feat_dim=24)
        self.action_encoder = ActionEncoder(d_model=d_model)
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh() # Outputs [-1, 1]
        )
        
    def forward(self, state_inputs, action_inputs):
        """
        state_inputs: dict of tensors for the ObservationEncoder
            card_tokens, numeric_features, global_features, zone_masks
        action_inputs: dict of tensors for the ActionEncoder
        """
        # 1. Encode State
        # state_vector: (batch_size, d_model)
        state_vector = self.state_encoder(
            deck_id=state_inputs["deck_id"],
            card_tokens=state_inputs["card_tokens"],
            numeric_features=state_inputs["numeric_features"],
            global_features=state_inputs["global_features"],
            zone_masks=state_inputs["zone_masks"]
        )
        
        # 2. Encode Actions
        # action_vectors: (batch_size, num_options, d_model)
        action_vectors = self.action_encoder(action_inputs)
        
        # 3. Pointer Network Policy
        # We compute dot product: (batch_size, 1, d_model) @ (batch_size, d_model, num_options)
        state_vector_expanded = state_vector.unsqueeze(1) # (batch_size, 1, d_model)
        
        # Transpose action_vectors to (batch_size, d_model, num_options)
        action_vectors_t = action_vectors.transpose(1, 2)
        
        # logits: (batch_size, 1, num_options) -> (batch_size, num_options)
        logits = torch.bmm(state_vector_expanded, action_vectors_t).squeeze(1)
        
        # Mask out invalid actions if padding was used
        if "action_mask" in action_inputs:
            # action_mask: (batch_size, num_options) - 1 for valid, 0 for invalid
            # Add a large negative number to invalid logits
            mask = action_inputs["action_mask"]
            logits = logits + (1.0 - mask) * -1e9
            
        policy = torch.softmax(logits, dim=-1)
        
        # 4. Value
        value = self.value_head(state_vector).squeeze(-1) # (batch_size,)
        
        return logits, policy, value
