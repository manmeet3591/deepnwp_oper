import xarray as xr
import os
import numpy as np

def process_file(file_path, ds_era5, output_dir):
    ds_graphcast = xr.open_dataset(file_path)
    vars = ['t2m', 'tp06']
    times = ds_graphcast.time.values

    ds_graphcast_interp = ds_graphcast[vars].isel(history=0).interp(lat=ds_era5.latitude, lon=ds_era5.longitude)
    
    ds_era5_ = ds_era5.sel(time=times)
    ds_era5_['tp_graph'] = (('time', 'latitude', 'longitude'), ds_graphcast_interp.tp06.values)
    ds_era5_['t2m_graph'] = (('time', 'latitude', 'longitude'), ds_graphcast_interp.t2m.values)
    
    print(np.sum(np.isnan(ds_era5_.tp_graph.values)))
    
    output_file_path = os.path.join(output_dir, os.path.basename(file_path))
    ds_era5_.isel(time=slice(1,61)).to_netcdf(output_file_path)

def main(input_dir, output_dir, ds_era5):
    for filename in os.listdir(input_dir):
        if filename.endswith('.nc'):
            file_path = os.path.join(input_dir, filename)
            process_file(file_path, ds_era5, output_dir)
            print(f'Processed {filename}')

# Load the base dataset (ds_era5) only once if it does not change
ds_era5 = xr.open_dataset('/scratch/08105/ms86336/graphcast/era5/tp/era5_2021_6hr_sum.nc')

# Define the input and output directories
input_dir = '/scratch/08105/ms86336/graphcast/2021'
output_dir = '/scratch/08105/ms86336/training/'

# Run the main processing function
main(input_dir, output_dir, ds_era5)

