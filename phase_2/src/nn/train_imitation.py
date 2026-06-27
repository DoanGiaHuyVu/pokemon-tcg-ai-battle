import torch
import torch.nn as nn
import torch.optim as optim
import os
from dataset import get_dataloader
from model import PointerPolicyModel

def train_imitation(data_path="data/imitation/dataset.jsonl", epochs=5, batch_size=32, lr=1e-3):
    print("Loading dataset...")
    dataloader = get_dataloader(data_path, batch_size=batch_size)
    
    print(f"Dataset size: {len(dataloader.dataset)}")
    
    model = PointerPolicyModel(d_model=128)
    
    # We will compute cross entropy over the valid options
    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        total_policy_loss = 0.0
        total_value_loss = 0.0
        correct_actions = 0
        total_actions = 0
        
        for state_inputs, action_inputs, target_action, value_target in dataloader:
            optimizer.zero_grad()
            
            logits, policy, value = model(state_inputs, action_inputs)
            
            # Policy loss
            p_loss = policy_criterion(logits, target_action)
            
            # Value loss
            v_loss = value_criterion(value, value_target)
            
            # Combined
            loss = p_loss + 0.1 * v_loss
            loss.backward()
            optimizer.step()
            
            total_policy_loss += p_loss.item()
            total_value_loss += v_loss.item()
            
            # Calculate accuracy
            preds = logits.argmax(dim=-1)
            correct_actions += (preds == target_action).sum().item()
            total_actions += target_action.size(0)
            
        avg_p_loss = total_policy_loss / len(dataloader)
        avg_v_loss = total_value_loss / len(dataloader)
        acc = correct_actions / total_actions * 100
        
        print(f"Epoch {epoch+1}/{epochs} | Policy Loss: {avg_p_loss:.4f} | Value Loss: {avg_v_loss:.4f} | Accuracy: {acc:.2f}%")
        
    print("Training complete! Model successfully learned the plumbing.")
    
    # Save the model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/imitation_v0.pt")
    print("Model saved to models/imitation_v0.pt")

if __name__ == "__main__":
    import sys
    data_path = sys.argv[1] if len(sys.argv) > 1 else "/app/data/imitation/dataset.jsonl"
    train_imitation(data_path)
