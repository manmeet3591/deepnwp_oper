import xarray as xr
import sys
file_path=sys.argv[1]
## Open the dataset
ds = xr.open_dataset(file_path)
#
## Check if 'time' is the correct dimension for resampling
#print(ds)
#
## Assuming 'tp' is in meters of water equivalent, and 'time' is the correct dimension
## Resample tp to 6-hourly sums
tp_6hr_sum = ds['tp'].resample(time='6H').sum()
tp_6hr_sum.to_netcdf(file_path[:-3]+'_6hr_sum.nc')
