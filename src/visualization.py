import xarray as xr

ds = xr.open_dataset('graphcast_2021_01_02_pred.nc').groupby('time.dayofyear').sum(dim='time') # make this better

tp_era5 = ds.tp * 1000
tp_graph = ds.tp_graph * 1000
tp_sfno = ds.tp_pred * 1000

tp_era5_clim = tp_era5.mean(dim='dayofyear')
tp_graph_clim = tp_graph.mean(dim='dayofyear')
tp_sfno_clim = tp_sfno.mean(dim='dayofyear')

tp_era5_std = tp_era5.std(dim='dayofyear')
tp_graph_std = tp_graph.std(dim='dayofyear')
tp_sfno_std = tp_sfno.std(dim='dayofyear')

# Compute anomalies
tp_era5_anomaly = tp_era5 #  (tp_era5 - tp_era5_clim) / (tp_era5_std)
tp_graph_anomaly =  (((tp_graph - tp_graph_clim) / (tp_graph_std))*tp_era5_std)+tp_era5_clim
tp_sfno_anomaly =   (((tp_sfno - tp_sfno_clim) / (tp_sfno_std))*tp_era5_std)+tp_era5_clim

tp_graph_anomaly.sel(latitude=slice(40,5)).sel(longitude=slice(70,100)).isel(dayofyear=1).plot(cmap='Blues', vmin=0, vmax=50)

tp_sfno_anomaly.sel(latitude=slice(40,5)).sel(longitude=slice(70,100)).isel(dayofyear=1).plot(cmap='Blues', vmin=0, vmax=50)

tp_era5_anomaly.sel(latitude=slice(40,5)).sel(longitude=slice(70,100)).isel(dayofyear=1).plot(cmap='Blues', vmin=0, vmax=50)

tp_graph_anomaly.sel(latitude=slice(30,20)).sel(longitude=slice(85,100)).mean(dim='latitude').mean(dim='longitude').plot()

tp_sfno_anomaly.sel(latitude=slice(30,20)).sel(longitude=slice(85,100)).mean(dim='latitude').mean(dim='longitude').plot()

tp_era5_anomaly.sel(latitude=slice(30,20)).sel(longitude=slice(85,100)).mean(dim='latitude').mean(dim='longitude').plot()



