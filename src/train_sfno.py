import torch

# Check if CUDA (GPU support) is available
gpu_available = torch.cuda.is_available()

print("GPU Available:", gpu_available)

# If a GPU is available, print the GPU name
if gpu_available:
    print("GPU Name:", torch.cuda.get_device_name(0))
import xarray as xr
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os
import glob
from tqdm import tqdm
# Constants for normalization
MIN_TP_GRAPH = -0.0022 # your_min_value_here
MAX_TP_GRAPH = 0.156354  # your_max_value_here
MIN_T2M_GRAPH = 189.5264  # your_min_value_here
MAX_T2M_GRAPH = 325.1456  # your_max_value_here
MIN_TP = 0  # your_min_value_here
MAX_TP = 0.3346  # your_max_value_here

class ncDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __getitem__(self, index):
        x = torch.from_numpy(self.data[index]).float().unsqueeze(0)
        y = torch.from_numpy(self.targets[index]).float().unsqueeze(0)
        return x, y

    def __len__(self):
        return len(self.data)

def normalize(data, min_val, max_val):
    return (data - min_val) / (max_val - min_val)
def normalize_gpu(data, min_val, max_val):
    """
    Normalize the data using the min and max values, performed on the GPU.
    
    Parameters:
    data (Tensor): Input tensor.
    min_val (float): Minimum value for normalization.
    max_val (float): Maximum value for normalization.
    
    Returns:
    Tensor: Normalized data.
    """
    # Ensure the data is a tensor and is on the GPU
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data)
    if not data.is_cuda:
        data = data.to('cuda')

    # Perform normalization on the GPU
    return (data - min_val) / (max_val - min_val)

def load_and_preprocess_data(folder_path):
    # Find all NetCDF files in the specified folder
    files = glob.glob(os.path.join(folder_path, '*.nc'))
    data_list = []
    target_list = []
    print('...Reading files...')
    for file in tqdm(files):
        print(file) #ds = xr.open_dataset(file)#.load()
        ds = xr.open_dataset(file, chunks={"time": 10})#.load()
        # Normalize the data
        tp_graph_norm = normalize_gpu(ds['tp_graph'].values, MIN_TP_GRAPH, MAX_TP_GRAPH)
        t2m_graph_norm = normalize_gpu(ds['t2m_graph'].values, MIN_T2M_GRAPH, MAX_T2M_GRAPH)
        tp_norm = normalize_gpu(ds['tp'].values, MIN_TP, MAX_TP)
        
        # Stack inputs along a new channel dimension and add to list
        inputs = torch.stack([tp_graph_norm, t2m_graph_norm], axis=0)  # Adding channel dimension
        data_list.append(inputs)
        target_list.append(tp_norm)
        
    # Concatenate all data into single arrays
    data_array = torch.concatenate(data_list, axis=1)  # Check axis based on your data structure
    target_array = torch.concatenate(target_list, axis=0)
    
    return data_array, target_array

# Main program execution
folder_path = '/scratch/08105/ms86336/training/'
data, targets = load_and_preprocess_data(folder_path)

# Split data into training and validation
X_train, X_val, y_train, y_val = train_test_split(data, targets, test_size=0.2, random_state=42)

# Create dataset objects
train_dataset = ncDataset(X_train, y_train)
val_dataset = ncDataset(X_val, y_val)

# Example of how to create a DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Now you can use train_loader and val_loader in your model training loop
lr, hr = train_dataset.__getitem__(0)
print(lr.shape, hr.shape, train_dataset.__len__())
