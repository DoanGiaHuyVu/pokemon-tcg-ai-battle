import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.nn.dataset_belief import ImitationDataset
from torch.utils.data import DataLoader
from src.nn.model_belief import PointerPolicyModel

def train_imitation(data_path="data/imitation/dataset_large.jsonl", epochs=10, batch_size=64, lr=1e-3):
    print("Loading datasets...")
    train_dataset = ImitationDataset(data_path, split="train")
    val_dataset = ImitationDataset(data_path, split="val")
    
    print(f"Train size: {len(train_dataset)} | Val size: {len(val_dataset)}")
    
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("Dataset is too small to train.")
        return
        
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device("cpu")
    model = PointerPolicyModel(d_model=128).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()
    
    print("Starting v1 training...")
    for epoch in range(epochs):
        model.train()
        total_p_loss, total_v_loss = 0, 0
        
        for state_inputs, action_inputs, target_action, value_target in train_loader:
            state_inputs = {k: v.to(device) for k, v in state_inputs.items()}
            action_inputs = {k: v.to(device) for k, v in action_inputs.items()}
            target_action = target_action.to(device)
            value_target = value_target.to(device, dtype=torch.float32)
            
            optimizer.zero_grad()
            logits, _, value = model(state_inputs, action_inputs)
            
            # Mask invalid options
            mask = action_inputs["action_mask"]
            logits = logits + (1.0 - mask) * -1e9
            
            p_loss = policy_criterion(logits, target_action)
            v_loss = value_criterion(value, value_target)
            loss = p_loss + v_loss
            
            loss.backward()
            optimizer.step()
            
            total_p_loss += p_loss.item()
            total_v_loss += v_loss.item()
            
        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for state_inputs, action_inputs, target_action, _ in val_loader:
                state_inputs = {k: v.to(device) for k, v in state_inputs.items()}
                action_inputs = {k: v.to(device) for k, v in action_inputs.items()}
                
                logits, _, _ = model(state_inputs, action_inputs)
                mask = action_inputs["action_mask"]
                logits = logits + (1.0 - mask) * -1e9
                
                preds = logits.argmax(dim=-1)
                correct += (preds == target_action.to(device)).sum().item()
                total += target_action.size(0)
                
        val_acc = (correct / total) * 100 if total > 0 else 0
        print(f"Epoch {epoch+1}/{epochs} | Train P-Loss: {total_p_loss/len(train_loader):.4f} | Train V-Loss: {total_v_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2f}%")
        
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/imitation_v1_belief.pt")
    print("Model saved to models/imitation_v1_belief.pt")

if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else "/app/data/imitation/dataset_large.jsonl"
    train_imitation(data_path)
