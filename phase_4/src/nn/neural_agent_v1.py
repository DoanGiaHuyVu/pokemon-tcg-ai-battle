import os
import sys
import torch

# Ensure cg is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
sys.path.append(os.path.dirname(__file__))

from model import PointerPolicyModel
from dataset import ImitationDataset
import agents.dragapult_agent as fallback_agent

# Use Dragapult's deck for the Neural Agent
my_deck = fallback_agent.my_deck

# Load Model Weights globally so we don't reload on every agent step
model_path = os.path.join(os.path.dirname(__file__), "../../models/imitation_v1.pt")
device = torch.device("cpu")
model = PointerPolicyModel(d_model=128).to(device)

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded Neural Policy from {model_path}")
else:
    print(f"Warning: Model not found at {model_path}. Using untrained random weights.")

# Instantiate an empty parser to reuse feature extraction logic
parser = ImitationDataset(None)
parser.data = []  # Empty

def agent(obs: dict) -> list[int] | int:
    """
    Neural Policy Agent.
    Evaluates the board, scores legal options, and picks the highest scoring one.
    Falls back to heuristic if an error occurs.
    """
    if "select" not in obs or obs["select"] is None:
        return [0]
        
    options = obs["select"].get("option", [])
    if not options:
        return [0]
        
    try:
        # Multi-selection is not yet supported by our simple pointer network
        if obs["select"].get("maxCount", 1) > 1:
            return fallback_agent.agent(obs)
            
        # 1. Parse Observation using Dataset Logic
        your_index = obs.get("current", {}).get("yourIndex", 0)
        
        global_features = parser._parse_global_features(obs.get("current")).unsqueeze(0)
        card_tokens, numeric_features, zone_masks = parser._parse_cards(obs.get("current"), your_index)
        
        state_inputs = {
            "card_tokens": card_tokens.unsqueeze(0),
            "numeric_features": numeric_features.unsqueeze(0),
            "global_features": global_features,
            "zone_masks": zone_masks.unsqueeze(0)
        }
        
        # 2. Parse Actions
        action_inputs = parser._parse_actions(options)
        # Add batch dimension
        for k in action_inputs:
            action_inputs[k] = action_inputs[k].unsqueeze(0)
            
        # 3. Model Inference
        with torch.no_grad():
            logits, policy, value = model(state_inputs, action_inputs)
            
        # 4. Action Selection
        mask = action_inputs["action_mask"][0]
        scores = logits[0]
        
        valid_indices = mask.nonzero(as_tuple=True)[0]
        if len(valid_indices) == 0:
            return fallback_agent.agent(obs)
            
        valid_scores = scores[valid_indices]
        sorted_valid = valid_indices[valid_scores.argsort(descending=True)].tolist()
        
        min_count = obs["select"].get("minCount", 1)
        max_count = obs["select"].get("maxCount", 1)
        
        if min_count == max_count:
            k = max_count
        else:
            # Naive heuristic: always take maximum allowed to maximize resources
            k = max_count
            
        k = min(k, len(sorted_valid))
        selected = sorted_valid[:k]
        
        if len(selected) < min_count:
            selected = sorted_valid[:min_count]
            
        # Fallback safeguard against simulator crashes during multi-select if out of bounds
        for s in selected:
            if s >= len(options):
                print(f"[Neural Agent] WARNING: index {s} >= len(options) {len(options)}")
                return fallback_agent.agent(obs)
                
        return selected
        
    except Exception as e:
        print(f"[Neural Agent Exception] {e}. Falling back to heuristic.")
        return fallback_agent.agent(obs)

if __name__ == "__main__":
    from cg.game import battle_start, battle_select
    import agents.random_agent as opponent_agent
    
    deck_0 = my_deck
    try:
        deck_1 = opponent_agent.my_deck
    except AttributeError:
        deck_1 = opponent_agent.read_deck_csv()
    
    obs_dict, _ = battle_start(deck_0, deck_1)
    
    step_count = 0
    while True:
        if obs_dict.get("current") and obs_dict["current"]["result"] != -1:
            print(f"Neural Agent vs Random finished in {step_count} steps. Winner: {obs_dict['current']['result']}")
            break
            
        if obs_dict.get("current")["yourIndex"] == 0:
            action = agent(obs_dict)
        else:
            action = opponent_agent.agent(obs_dict)
            
        obs_dict = battle_select(action)
        step_count += 1
        
        if step_count > 1000:
            print("Match timed out.")
            break
