import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.nn.dataset_v3 import ImitationDataset
from src.nn.model_v3 import PointerPolicyModel
from src.nn.card_metadata_features import generate_card_metadata

def collate_fn(batch):
    # Batch is a list of dicts: [{"state": {}, "actions": {}, "target": t}]
    state_keys = batch[0]["state"].keys()
    action_keys = batch[0]["actions"].keys()
    
    batched_state = {}
    for k in state_keys:
        batched_state[k] = torch.stack([item["state"][k] for item in batch])
        
    batched_actions = {}
    for k in action_keys:
        batched_actions[k] = torch.stack([item["actions"][k] for item in batch])
        
    batched_targets = torch.stack([item["target"] for item in batch])
    
    return batched_state, batched_actions, batched_targets

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    dataset_expert_path = os.path.join(os.path.dirname(__file__), "../../../data/dataset_v3_expert.jsonl")
    dataset_beam_path = os.path.join(os.path.dirname(__file__), "../../../data/dataset_v3_beam.jsonl")
    dataset_paths = [dataset_expert_path, dataset_beam_path]
        
    train_dataset = ImitationDataset(dataset_paths, split="train")
    val_dataset = ImitationDataset(dataset_paths, split="val")
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
    
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    
    model = PointerPolicyModel(d_model=128).to(device)
    
    # Load and inject static card metadata
    print("Generating Card Metadata...")
    # This requires cg to be initialized (lib.AllCard())
    # We will assume it works if cg.api is loadable.
    # Actually, generate_card_metadata() requires the C++ library, which must run in docker.
    metadata_tensor = generate_card_metadata().to(device)
    model.state_encoder.load_metadata(metadata_tensor)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion_policy = nn.CrossEntropyLoss(ignore_index=-1)
    
    epochs = 3
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (states, actions, targets) in enumerate(train_loader):
            states = {k: v.to(device) for k, v in states.items()}
            actions = {k: v.to(device) for k, v in actions.items()}
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            logits, policy, value = model(states, actions)
            
            # Policy Loss
            loss = criterion_policy(logits, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            preds = logits.argmax(dim=1)
            train_correct += (preds == targets).sum().item()
            train_total += targets.size(0)
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
                
        train_acc = train_correct / max(1, train_total)
        
        # Eval
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for states, actions, targets in val_loader:
                states = {k: v.to(device) for k, v in states.items()}
                actions = {k: v.to(device) for k, v in actions.items()}
                targets = targets.to(device)
                
                logits, policy, value = model(states, actions)
                loss = criterion_policy(logits, targets)
                
                val_loss += loss.item()
                preds = logits.argmax(dim=1)
                val_correct += (preds == targets).sum().item()
                val_total += targets.size(0)
                
        val_acc = val_correct / max(1, val_total)
        print(f"Epoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss/max(1, len(train_loader)):.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss/max(1, len(val_loader)):.4f} | Val Acc: {val_acc:.4f}")
        
    os.makedirs(os.path.join(os.path.dirname(__file__), "../../models"), exist_ok=True)
    save_path = os.path.join(os.path.dirname(__file__), "../../models/imitation_v3.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Saved model to {save_path}")

if __name__ == "__main__":
    train()
