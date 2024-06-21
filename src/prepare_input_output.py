#import xarray as xr
#
#path = '/scratch/08105/ms86336/graphcast/2021/graphcast_2021_01_22.nc'
#
#ds = xr.open_dataset(path)
#vars = ['t2m', 'tp06']
#print(ds[vars].isel(history=0), ds.time.values[0])
#
#path = '/scratch/08105/ms86336/graphcast/era5/tp/'
#
#ds = xr.open_dataset(path+'era5_2021.nc')
#print(ds.tp.min(), ds.tp.max())


#import xarray as xr

# Path to the NetCDF file
#file_path = '/scratch/08105/ms86336/graphcast/era5/tp/era5_2021.nc'
#file_path = '/scratch/08105/ms86336/graphcast/era5/tp/era5_2022.nc'
#
## Open the dataset
#ds = xr.open_dataset(file_path)
#
## Check if 'time' is the correct dimension for resampling
#print(ds)
#
## Assuming 'tp' is in meters of water equivalent, and 'time' is the correct dimension
## Resample tp to 6-hourly sums
#tp_6hr_sum = ds['tp'].resample(time='6H').sum()
#
## Print the resampled data
#print(tp_6hr_sum)
#
## Close the dataset
#ds.close()
#
## Optionally, save the resampled data to a new NetCDF file
##tp_6hr_sum.to_netcdf('/scratch/08105/ms86336/graphcast/era5/tp/era5_2021_6hr_sum.nc')
#tp_6hr_sum.to_netcdf('/scratch/08105/ms86336/graphcast/era5/tp/era5_2022_6hr_sum.nc')

import xarray as xr
import numpy as np
ds_era5 = xr.open_dataset('/scratch/08105/ms86336/graphcast/era5/tp/era5_2021_6hr_sum.nc')

#print(ds_era5)

path = '/scratch/08105/ms86336/graphcast/2021/graphcast_2021_01_22.nc'

ds_graphcast = xr.open_dataset(path)
vars = ['t2m', 'tp06']
#print(ds_graphcast[vars].isel(history=0), ds_graphcast.time.values[0])

times = ds_graphcast.time.values

#print('Time = ', times)

#print(ds_era5.sel(time=times))
#print(ds_era5.sel(time=times).time.values)

ds_graphcast_interp = ds_graphcast[vars].isel(history=0).interp(lat=ds_era5.latitude, lon=ds_era5.longitude)

print(ds_graphcast_interp)

print(ds_era5.sel(time=times))

ds_era5_ = ds_era5.sel(time=times)
ds_era5_['tp_graph'] = (('time', 'latitude', 'longitude'), ds_graphcast_interp.tp06.values)
ds_era5_['t2m_graph'] = (('time', 'latitude', 'longitude'), ds_graphcast_interp.t2m.values)
print(ds_era5_)

print(np.sum(np.isnan(ds_era5_.tp_graph.values)))
print(ds_era5_.isel(time=slice(1,61)))
ds_era5_.isel(time=slice(1,61)).to_netcdf('test.nc')

