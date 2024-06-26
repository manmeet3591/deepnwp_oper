import torch
import xarray as xr
import h5py
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import sys
from sfno_embedding import *
import sys

# Constants for normalization
MIN_TP_GRAPH = -0.0022
MAX_TP_GRAPH = 0.156354
MIN_T2M_GRAPH = 189.5264
MAX_T2M_GRAPH = 325.1456
MIN_TP = 0
MAX_TP = 0.3346

# Function to normalize data
def normalize_data(inputs):
    min_val_input = {'tp_graph': MIN_TP_GRAPH, 't2m_graph': MIN_T2M_GRAPH}
    max_val_input = {'tp_graph': MAX_TP_GRAPH, 't2m_graph': MAX_T2M_GRAPH}

    inputs[:60, ...] = (inputs[:60, ...] - min_val_input['tp_graph']) / (max_val_input['tp_graph'] - min_val_input['tp_graph'])
    inputs[60:, ...] = (inputs[60:, ...] - min_val_input['t2m_graph']) / (max_val_input['t2m_graph'] - min_val_input['t2m_graph'])

    return inputs
# Load the NetCDF file and convert to HDF5
netcdf_file = sys.argv[1]  #'/path/to/your/netcdf_file.nc'
hdf5_file = netcdf_file.replace('.nc', '.h5')

# Convert NetCDF to HDF5
with xr.open_dataset(netcdf_file) as ds:
    with h5py.File(hdf5_file, 'w') as hdf:
        for var_name in ['tp_graph', 't2m_graph', 'tp']:
            if var_name in ds:
                data_array = ds[var_name]
                hdf.create_dataset(var_name, data=data_array.values, dtype='float32')


with h5py.File(hdf5_file, 'r') as file:
    tp_graph = file['tp_graph'][:]
    t2m_graph = file['t2m_graph'][:]

   
# Normalize each part separately
tp_graph = (tp_graph - MIN_TP_GRAPH) / (MAX_TP_GRAPH - MIN_TP_GRAPH)
t2m_graph = (t2m_graph - MIN_T2M_GRAPH) / (MAX_T2M_GRAPH - MIN_T2M_GRAPH)

# Concatenate and reshape
inputs = np.concatenate((tp_graph, t2m_graph), axis=0)
inputs = torch.from_numpy(inputs[np.newaxis]).float()

# Inference function
def inference(model, inputs, device):
    model.eval()
    #predictions = []

    with torch.no_grad():
        inputs = inputs.to(device)
        print('before inference')
        outputs = model(inputs)
        print('after inference')
        #predictions.append(outputs.cpu().numpy())

    return outputs.cpu().numpy() # np.concatenate(predictions, axis=0)

# Create the dataset and dataloader for inference
#sys.exit()
#dataloader = DataLoader(dataset, batch_size=20, shuffle=False)
#
#for batch in dataloader:
#    data = batch
#    print(data.size())  
#    break
#sys.exit()

# Load the model
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
model = SphericalFourierNeuralOperatorNet(
    spectral_transform='sht', 
    operator_type='driscoll-healy', 
    img_size=(721, 1440), 
    in_chans=120, 
    out_chans=60, 
    grid="equiangular",
    num_layers=4, 
    scale_factor=3, 
    embed_dim=16, 
    big_skip=True, 
    pos_embed="lat", 
    use_mlp=False, 
    normalization_layer="none"
).to(device)

model_save_path = "/scratch/08105/ms86336/best_model_sfno.pth"
if os.path.exists(model_save_path):
    model.load_state_dict(torch.load(model_save_path))
    print(f"Loaded model from {model_save_path}")
print(inputs.shape)
# Perform inference
tp_pred = inference(model, inputs, device)
print('Shape of the predicted array: ', tp_pred.shape)  # Check the shape of the predictions

tp_pred_inverse_normalization = tp_pred * (MAX_TP - MIN_TP) + MIN_TP

ds_pred = xr.open_dataset(netcdf_file)
ds_pred['tp_pred'] = (('time', 'latitude', 'longitude'), tp_pred_inverse_normalization[0,:,:,:])

# Save tp_pred to a new NetCDF file
#ds_pred = xr.Dataset({'tp_pred': (('time', 'lat', 'lon'), tp_pred)})
ds_pred.to_netcdf(netcdf_file[:-3]+'_pred.nc')

