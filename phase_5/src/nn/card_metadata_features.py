import os
import sys
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from cg.api import all_card_data, CardType, EnergyType

def generate_card_metadata():
    """
    Parses Kaggle's all_card_data() into a static PyTorch tensor.
    Returns a tensor of shape (MAX_CARD_ID + 1, num_features).
    """
    cards = all_card_data()
    max_id = max([c.cardId for c in cards])
    
    # Define features
    # 0: is_pokemon
    # 1: is_energy
    # 2: is_trainer
    # 3: hp (normalized, max hp is ~340)
    # 4: retreat_cost (max ~4)
    # 5: is_basic
    # 6: is_stage1
    # 7: is_stage2
    # 8: is_ex / mega_ex
    # 9: attack1_damage (normalized)
    # 10: attack2_damage (normalized)
    
    num_features = 11
    feature_matrix = torch.zeros((max_id + 1, num_features), dtype=torch.float32)
    
    for c in cards:
        idx = c.cardId
        
        # Types
        if c.cardType == CardType.POKEMON:
            feature_matrix[idx, 0] = 1.0
        elif c.cardType in [CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY]:
            feature_matrix[idx, 1] = 1.0
        else: # Supporter, Item, Stadium, Tool
            feature_matrix[idx, 2] = 1.0
            
        # Stats
        feature_matrix[idx, 3] = c.hp / 340.0
        feature_matrix[idx, 4] = min(c.retreatCost, 4) / 4.0
        
        # Stages
        if getattr(c, "basic", False): feature_matrix[idx, 5] = 1.0
        if getattr(c, "stage1", False): feature_matrix[idx, 6] = 1.0
        if getattr(c, "stage2", False): feature_matrix[idx, 7] = 1.0
        
        # EX rules
        if getattr(c, "ex", False) or getattr(c, "megaEx", False):
            feature_matrix[idx, 8] = 1.0
            
        # Attacks
        if hasattr(c, "attacks") and c.attacks:
            # We can't query attack damage easily without cg.api.Attack parsing,
            # but wait, we have to look up the attack details.
            pass
            
    return feature_matrix

if __name__ == "__main__":
    matrix = generate_card_metadata()
    print(f"Generated card metadata matrix of shape {matrix.shape}")
    print(f"Example Card ID 1: {matrix[1].tolist()}")
