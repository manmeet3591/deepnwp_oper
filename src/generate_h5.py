import h5py
import torch
from torch.utils.data import DataLoader, Dataset
import glob
import sys
from sfno_embedding import *
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, ToTensor, Resize
from PIL import Image
import os
import numpy as np
import sys
# Constants for normalization
MIN_TP_GRAPH = -0.0022 # your_min_value_here
MAX_TP_GRAPH = 0.156354  # your_max_value_here
MIN_T2M_GRAPH = 189.5264  # your_min_value_here
MAX_T2M_GRAPH = 325.1456  # your_max_value_here
MIN_TP = 0  # your_min_value_here
MAX_TP = 0.3346  # your_max_value_here

def normalize_data(inputs, tp):
    # Example normalization: adjust based on actual data ranges
    min_val_input = {'tp_graph': MIN_TP_GRAPH, 't2m_graph': MIN_T2M_GRAPH}
    max_val_input = {'tp_graph': MAX_TP_GRAPH, 't2m_graph': MAX_T2M_GRAPH}
    min_val_tp = MIN_TP
    max_val_tp = MAX_TP

    inputs[:60, ...] = (inputs[:60, ...] - min_val_input['tp_graph']) / (max_val_input['tp_graph'] - min_val_input['tp_graph'])
    
    # Normalize t2m_graph part
    inputs[60:, ...] = (inputs[60:, ...] - min_val_input['t2m_graph']) / (max_val_input['t2m_graph'] - min_val_input['t2m_graph'])

    # Normalize tp
    tp = (tp - min_val_tp) / (max_val_tp - min_val_tp)

    return inputs, tp

#    # Normalize
#    inputs[0, ...] = (inputs[0, ...] - min_val_input['tp_graph']) / (max_val_input['tp_graph'] - min_val_input['tp_graph'])
#    inputs[1, ...] = (inputs[1, ...] - min_val_input['t2m_graph']) / (max_val_input['t2m_graph'] - min_val_input['t2m_graph'])
#    tp = (tp - min_val_tp) / (max_val_tp - min_val_tp)
#
#    return inputs, tp

class LargeHDF5Dataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Load data from a single HDF5 file
        with h5py.File(self.file_paths[idx], 'r') as file:
            tp_graph = torch.from_numpy(file['tp_graph'][:]).float()
            t2m_graph = torch.from_numpy(file['t2m_graph'][:]).float()
            tp = torch.from_numpy(file['tp'][:]).float()
        # Stack the channels and return
        inputs = torch.cat([tp_graph, t2m_graph], dim=0)
        return inputs, tp

# Get all HDF5 files
hdf_files = glob.glob('/scratch/08105/ms86336/training/*.h5')
val_files = glob.glob('/scratch/08105/ms86336/training/validation/*.h5')

# Create dataset
train_dataset = LargeHDF5Dataset(hdf_files, transform=normalize_data)
val_dataset = LargeHDF5Dataset(val_files, transform=normalize_data)

#inputs, tp = dataset.__getitem__(0)
#print(inputs.shape, tp.shape)
#
#sys.exit()
# DataLoader for managing batches
train_dataloader = DataLoader(train_dataset, batch_size=20, shuffle=True)  # Adjust batch size based on your memory and needs
val_dataloader = DataLoader(val_dataset, batch_size=20, shuffle=True)  # Adjust batch size based on your memory and needs

for batch in train_dataloader:
    data, targets = batch
    print(data.size())  # Should print torch.Size([16, 1, 30, 30])
    print(targets.size())  # Should print torch.Size([16, 1, 30, 601])
    break

#sys.exit()
## Process each batch
#for batch_idx, (inputs, targets) in enumerate(loader):
#    print(f"Processing batch {batch_idx + 1}")
#    # Process your data here: e.g., training a model
#    # model.train(inputs, targets)
#

def train(model, train_dataloader, val_dataloader, criterion, optimizer, device):
    model.train()
    train_loss = 0.0
    for batch in train_dataloader:
        lr, hr = batch
        lr, hr = lr.to(device), hr.to(device)
        optimizer.zero_grad()
        sr = model(lr)
        loss = criterion(sr, hr)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_dataloader)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_dataloader:
            lr, hr = batch
            lr, hr = lr.to(device), hr.to(device)
            sr = model(lr)
            loss = criterion(sr, hr)
            val_loss += loss.item()

    val_loss /= len(val_dataloader)

    return train_loss, val_loss

# Initialize the model, loss function, and optimizer
device = 'cuda'

nlat = 721
nlon = 1440
model = SphericalFourierNeuralOperatorNet(spectral_transform='sht', operator_type='driscoll-healy', img_size=(nlat, nlon), in_chans=120, out_chans=60, grid="equiangular",
                                          num_layers=4, scale_factor=3, embed_dim=16, big_skip=True, pos_embed="lat", use_mlp=False, normalization_layer="none").to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
model_save_path = "/scratch/08105/ms86336/best_model_sfno.pth"
if os.path.exists(model_save_path):
    model.load_state_dict(torch.load(model_save_path))
    print(f"Loaded model from {model_save_path}")
from copy import deepcopy

num_epochs = 100000
print_interval = 10
patience = 1000
best_val_loss = float('inf')
counter = 0
best_model = None

with tqdm(total=num_epochs, desc="Training Progress") as pbar:
    for epoch in range(1, num_epochs + 1):
#for epoch in tqdm(range(1, num_epochs + 1), desc="Training Progress"):
#for epoch in range(1, num_epochs + 1):
        train_loss, val_loss = train(model, train_dataloader, val_dataloader, criterion, optimizer, device)
    # Log losses to TensorBoard
    
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            #best_model = deepcopy(model)
            counter = 0
            torch.save(model.state_dict(), model_save_path)
        else:
            counter += 1
        
        if epoch % print_interval == 0:
            print(f"Epoch [{epoch}/{num_epochs}] - Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        # Update tqdm with the current loss values
        pbar.set_postfix_str(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        pbar.update()
        
        if counter >= patience:
            print("Early stopping triggered.")
            break

