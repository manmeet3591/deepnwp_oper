import h5py
import torch
from torch.utils.data import DataLoader, Dataset
import glob
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

    # Normalize
    inputs[0, ...] = (inputs[0, ...] - min_val_input['tp_graph']) / (max_val_input['tp_graph'] - min_val_input['tp_graph'])
    inputs[1, ...] = (inputs[1, ...] - min_val_input['t2m_graph']) / (max_val_input['t2m_graph'] - min_val_input['t2m_graph'])
    tp = (tp - min_val_tp) / (max_val_tp - min_val_tp)

    return inputs, tp

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
        inputs = torch.stack([tp_graph, t2m_graph], dim=0)
        return inputs, tp

# Get all HDF5 files
hdf_files = glob.glob('/scratch/08105/ms86336/training/*.h5')

# Create dataset
dataset = LargeHDF5Dataset(hdf_files, transform=normalize_data)

#inputs, tp = dataset.__getitem__(0)
#print(inputs.shape, tp.shape)
#
#sys.exit()
# DataLoader for managing batches
loader = DataLoader(dataset, batch_size=50, shuffle=True)  # Adjust batch size based on your memory and needs

# Process each batch
for batch_idx, (inputs, targets) in enumerate(loader):
    print(f"Processing batch {batch_idx + 1}")
    # Process your data here: e.g., training a model
    # model.train(inputs, targets)

