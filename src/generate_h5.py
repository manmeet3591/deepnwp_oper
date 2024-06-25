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
import sys

# Constants for normalization
MIN_TP_GRAPH = -0.0022 # your_min_value_here
MAX_TP_GRAPH = 0.156354  # your_max_value_here
MIN_T2M_GRAPH = 189.5264  # your_min_value_here
MAX_T2M_GRAPH = 325.1456  # your_max_value_here
MIN_TP = 0  # your_min_value_here
MAX_TP = 0.3346  # your_max_value_here
generate_h5 = False

import xarray as xr
import h5py
import numpy as np


import os
import glob
import h5py
import xarray as xr

folder_path = '/scratch/08105/ms86336/training/'
netcdf_files = glob.glob(os.path.join(folder_path, '*.nc'))

if generate_h5:

    # Create an HDF5 file for each NetCDF file
    for nc_file in netcdf_files:
        with xr.open_dataset(nc_file) as ds:
            hdf_file_path = nc_file.replace('.nc', '.h5')
            with h5py.File(hdf_file_path, 'w') as hdf:
                for var_name in ['tp_graph', 't2m_graph', 'tp']:
                    if var_name in ds:
                        data_array = ds[var_name]
                        hdf.create_dataset(var_name, data=data_array.values, dtype='float32')
    
import h5py
import numpy as np

hdf_files = glob.glob('/scratch/08105/ms86336/training/*.h5')
output_file = 'combined_data.h5'

with h5py.File(output_file, 'w') as output_hdf:
    # Prepare dictionaries to hold data temporarily
    data_holder = {var_name: [] for var_name in ['tp_graph', 't2m_graph', 'tp']}

    # Load data from each HDF5 file
    for hdf_file in hdf_files:
        with h5py.File(hdf_file, 'r') as input_hdf:
            for var_name in data_holder.keys():
                if var_name in input_hdf:
                    data_holder[var_name].append(input_hdf[var_name][:])

    # Stack and save each variable into the new HDF5 file
    for var_name, data_list in data_holder.items():
        stacked_data = np.stack(data_list, axis=0)
        output_hdf.create_dataset(var_name, data=stacked_data, dtype='float32')
import h5py
import numpy as np

hdf_files = glob.glob('/scratch/08105/ms86336/training/*.h5')


import os
import h5py
import torch
from torch.utils.data import Dataset

class MultiHDF5Dataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform
        self.file_lengths = []
        self.total_length = 0

        # Calculate the total length and per-file lengths
        for path in self.file_paths:
            with h5py.File(path, 'r') as file:
                length = file['tp'].shape[0]
                self.file_lengths.append(length)
                self.total_length += length

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        # Determine which file and which local index this corresponds to
        file_index, local_index = self.find_file_index(idx)

        # Read data
        with h5py.File(self.file_paths[file_index], 'r') as file:
            print('Reading data from file')
            tp_graph = torch.from_numpy(file['tp_graph'][local_index]).float()
            t2m_graph = torch.from_numpy(file['t2m_graph'][local_index]).float()
            tp = torch.from_numpy(file['tp'][local_index]).float()

            # Stack the input channels
            inputs = torch.stack([tp_graph, t2m_graph], dim=0)

            # Normalize data
            if self.transform:
                inputs, tp = self.transform(inputs, tp)

        return inputs, tp

    def find_file_index(self, global_index):
        # Iterate over files to find the correct one
        cumulative_length = 0
        for i, length in enumerate(self.file_lengths):
            if cumulative_length + length > global_index:
                # Return file index and local index within the file
                return i, global_index - cumulative_length
            cumulative_length += length

# Define a normalization transform function
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


from torch.utils.data import DataLoader

# List all your HDF5 files


dataset = MultiHDF5Dataset(hdf5_files, transform=normalize_data)

print(dataset.__getitem__(0))
print(dataset.__len__())

#folder_path = '/scratch/08105/ms86336/training/'
## Find all NetCDF files in the specified folder
#netcdf_files = glob.glob(os.path.join(folder_path, '*.nc'))
## Define the chunk size for each operation (choose a size that fits comfortably in your memory)
#chunk_size = 10  # This depends on your system's memory; adjust as needed
#
## Determine the total number of chunks
#num_chunks = len(netcdf_files) // chunk_size + (0 if len(netcdf_files) % chunk_size == 0 else 1)
#
## Prepare to create an HDF5 file
#with h5py.File('combined_data.h5', 'w') as hdf:
#    # Initialize datasets within the HDF5 file with appropriate dimensions
#    # Here, you would need to know the dimensions beforehand or calculate them
#    # Let's assume the dimensions are known: (730, 15, 720, 1440) for x1 and x2, (730, 720, 1440) for y
#    x1_dset = hdf.create_dataset('x1', (730, 15, 721, 1440), dtype='float32')
#    x2_dset = hdf.create_dataset('x2', (730, 15, 721, 1440), dtype='float32')
#    y_dset = hdf.create_dataset('y', (730, 15, 721, 1440), dtype='float32')
#
#    start_idx = 0
#    for chunk_idx in range(num_chunks):
#        # Calculate the range of files to load in this chunk
#        start_file = chunk_idx * chunk_size
#        end_file = min((chunk_idx + 1) * chunk_size, len(netcdf_files))
#        
#        # Load a chunk of NetCDF files
#        chunk_files = netcdf_files[start_file:end_file]
#        dataset = xr.open_mfdataset(chunk_files, concat_dim='time', combine='nested', 
#                                    data_vars=['tp_graph', 't2m_graph', 'tp'])
#
#        # Calculate end index for this chunk in the HDF5 dataset
#        end_idx = start_idx + dataset.dims['time']
#
#        # Write the data to HDF5
#        x1_dset[start_idx:end_idx, :, :, :] = dataset['tp_graph'].values
#        x2_dset[start_idx:end_idx, :, :, :] = dataset['t2m_graph'].values
#        y_dset[start_idx:end_idx, :, :] = dataset['tp'].values
#
#        # Update the start index for the next chunk
#        start_idx = end_idx
#
#        # Close the xarray dataset to free up memory
#        dataset.close()


sys.exit()

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
        data = data.to('cuda:0')

    # Perform normalization on the GPU
    return (data - min_val) / (max_val - min_val)

folder_path = '/scratch/08105/ms86336/training/'
# Find all NetCDF files in the specified folder
files = glob.glob(os.path.join(folder_path, '*.nc'))
data_list = []
target_list = []
print('...Reading files...')
for i in tqdm(range(0, len(files), 30)):
    batch_files = files[i:i+30]
    for file in batch_files:    
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
    
#    return data_array, target_array

# Main program execution
    data, targets = data_array, target_array

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
