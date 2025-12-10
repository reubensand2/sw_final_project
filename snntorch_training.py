import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import StandardScaler

import snntorch as snn
import snntorch.functional as SF

import numpy as np
import os

def process_batch(filename):
    print(f"Loading {filename}...")

    # load_svmlight_file automatically handles the "1:val 2:val" format
    X_sparse, y = load_svmlight_file(filename, n_features=128)
    
    # convert to numpy array
    X = X_sparse.toarray()

    # adjust labels to be 0-indexed
    y = y-1

    return X, y

def get_dataloaders(batch_files, batch_size=64):
    X_list, y_list = [], []
    
    # Load multiple files (e.g., batch1.dat, batch2.dat)
    for file in batch_files:
        if os.path.exists(file):
            X_b, y_b = process_batch(file)
            X_list.append(X_b)
            y_list.append(y_b)
        else:
            print(f"Warning: {file} not found. Skipping.")
            
    if not X_list:
        raise FileNotFoundError("No data files found.")

    # Concatenate all loaded batches
    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)

    # NORMALIZE (Crucial for SNNs to fire correctly)
    scaler = StandardScaler()
    X_all = scaler.fit_transform(X_all)

    # Save the constants
    np.save("scaler_mean.npy", scaler.mean_)
    np.save("scaler_scale.npy", scaler.scale_)

    print("Normalization parameters saved.")
    print(f"Mean[0]: {scaler.mean_[0]:.4f}, Scale[0]: {scaler.scale_[0]:.4f}")

    # Convert to Tensor
    tensor_x = torch.Tensor(X_all)
    tensor_y = torch.LongTensor(y_all)

    return DataLoader(TensorDataset(tensor_x, tensor_y), batch_size=batch_size, shuffle=True)

class GasSensorSNN(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs, beta=0.95):
        super().__init__()
        
        # Initialize layers
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self, x, num_steps=25):
        # Initialize hidden states (membrane potential)
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        
        # Record the final layer spikes
        spk2_rec = []
        
        # Time Loop
        for step in range(num_steps):
            # Layer 1
            cur1 = self.fc1(x) # Input x is constant (Direct Encoding)
            spk1, mem1 = self.lif1(cur1, mem1)
            
            # Layer 2
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            
            spk2_rec.append(spk2)
            
        return torch.stack(spk2_rec, dim=0)

if __name__ == "__main__":
    BATCH_FILES = ['./Dataset/batch2.dat', './Dataset/batch3.dat', './Dataset/batch6.dat']

    # 1. Load Data
    try:
        train_loader = get_dataloaders(BATCH_FILES)
        print("Data loaded successfully.")
    except Exception as e:
        print(f"Error loading data: {e}")
        exit()

    # 2. Instantiate Model
    model = GasSensorSNN(num_inputs=128, num_hidden=64, num_outputs=6)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    criterion = SF.ce_rate_loss()

    # 3. Training Loop
    epochs = 20
    print(f"Starting training on cpu...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for data, targets in train_loader:
            
            # Forward pass
            spk_rec = model(data)
            
            # Loss Calculation (Cross Entropy on Rate)
            loss_val: torch.Tensor = criterion(spk_rec, targets)
            
            # Gradient Step
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            
            # Accuracy Tracking
            total_loss += loss_val.item()
            # Sum spikes over time to get prediction index
            predicted = spk_rec.sum(dim=0).argmax(1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Acc: {100*correct/total:.2f}%")

    print("\nExporting weights for NumPy simulation...")
    
    # Extract weights and biases
    # We detatch from graph (.detach()) and move to CPU (.cpu()) and convert to numpy (.numpy())
    w1 = model.fc1.weight.detach().cpu().numpy().T # Transpose to shape [Input, Hidden]
    b1 = model.fc1.bias.detach().cpu().numpy()
    
    w2 = model.fc2.weight.detach().cpu().numpy().T # Transpose to shape [Hidden, Output]
    b2 = model.fc2.bias.detach().cpu().numpy()
    
    # Save as .npy files
    np.save("weights_fc1.npy", w1)
    np.save("bias_fc1.npy", b1)
    np.save("weights_fc2.npy", w2)
    np.save("bias_fc2.npy", b2)
    