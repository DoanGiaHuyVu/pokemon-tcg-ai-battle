import os
import sys
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.nn.model import PointerPolicyModel
from src.nn.dataset import ImitationDataset
from src.nn.card_metadata_features import generate_card_metadata
import agents.dragapult_agent as fallback_agent

my_deck = fallback_agent.my_deck

model_path = os.path.join(os.path.dirname(__file__), "../../models/imitation_v2.pt")

# Set device: use mps for Mac, cuda for Nvidia GPUs, else fallback to cpu
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    
model = PointerPolicyModel(d_model=128).to(device)

if os.path.exists(model_path):
    # Metadata must be loaded first
    try:
        metadata = generate_card_metadata()
        model.state_encoder.load_metadata(metadata)
    except Exception as e:
        print(f"Warning: Failed to load card metadata: {e}")
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded Neural Policy v2 from {model_path}")
else:
    print(f"Warning: Model not found at {model_path}. Using untrained random weights.")

parser = ImitationDataset(None)
parser.data = []

def agent(obs: dict) -> list[int] | int:
    """
    Neural Policy Agent v2.
    """
    if "select" not in obs or obs["select"] is None:
        return [0]
        
    options = obs["select"].get("option", [])
    if not options:
        return [0]
        
    try:
        if obs["select"].get("maxCount", 1) > 1:
            return fallback_agent.agent(obs)
            
        your_index = obs.get("current", {}).get("yourIndex", 0)
        
        global_features = parser._parse_global_features(obs.get("current")).unsqueeze(0).to(device)
        card_tokens, numeric_features, zone_masks = parser._parse_cards(obs.get("current"), your_index)
        
        # Determine deck id (hardcoded 0 for Dragapult for now, as my_deck is dragapult)
        deck_id = torch.tensor([0], dtype=torch.long).to(device)
        
        state_inputs = {
            "deck_id": deck_id,
            "card_tokens": card_tokens.unsqueeze(0).to(device),
            "numeric_features": numeric_features.unsqueeze(0).to(device),
            "global_features": global_features,
            "zone_masks": zone_masks.unsqueeze(0).to(device)
        }
        
        action_inputs = parser._parse_actions(options)
        for k in action_inputs:
            action_inputs[k] = action_inputs[k].unsqueeze(0).to(device)
            
        with torch.no_grad():
            logits, policy, value = model(state_inputs, action_inputs)
            
        mask = action_inputs["action_mask"][0]
        scores = logits[0]
        
        valid_indices = mask.nonzero(as_tuple=True)[0]
        if len(valid_indices) == 0:
            return fallback_agent.agent(obs)
            
        valid_scores = scores[valid_indices]
        sorted_valid = valid_indices[valid_scores.argsort(descending=True)].tolist()
        
        # V2 FIX: We finally support min_count = 0 correctly by allowing the empty list if score > threshold?
        # No, for now we still use Top-K logic, but the v2 search agent handles min_count=0 skipping!
        # This raw agent function is purely policy.
        min_count = obs["select"].get("minCount", 1)
        max_count = obs["select"].get("maxCount", 1)
        k = max_count
            
        k = min(k, len(sorted_valid))
        selected = sorted_valid[:k]
        
        if len(selected) < min_count:
            selected = sorted_valid[:min_count]
            
        return selected
        
    except Exception as e:
        print(f"Neural error v2: {e}")
        return fallback_agent.agent(obs)
