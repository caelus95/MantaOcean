# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 01:16:50 2021

@author: shamu
"""
temp= data.values
lon = data.lon_rho.values
lat = data.lat_rho.values
t = pd.date_range("1992-01-01", periods=len(data.ocean_time.values),freq='5D')

ds = xr.Dataset(
    {
        "temperature": (["time","x", "y"], temp),
    },
    coords={
        "lon": (["x", "y"], lon),
        "lat": (["x", "y"], lat),
        "time": t,
        "reference_time": pd.Timestamp("1992-01-01"),
    },
)


ds.temperature.sel(time=pd.date_range("1992-01-01",periods=1,freq='5D')).values

ds.temperature.sel(time=dt.datetime(1992,1,1)).values

Data_Mmean1 = ds.temperature.resample(time="1MS").mean(dim="time")


da.sel(time=dt.datetime(1992,1,1)).values


A= xr.open_dataset(w_path+data_name+'Mm1.nc')

