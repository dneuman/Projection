#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module that handles loading data from disk and supplying it to
requestors. It manages a dictionary of supplied data so that it
won't require reloading. Returns a copy of the data to remove
inter-routine problems. Data is safe to manipulate.

Common useage:
    import datastore as ds
    dst = ds.dst
    dt = dst.best()
"""

# %% Setup

import pandas as pd
import numpy as np
import requests
import yaml
from collections import namedtuple
from io import StringIO
import datetime as dt
import xarray as xr

yn, mn, dn, en, nn, sn = ['Year', 'Month', 'Data',
                          'Error', 'Normalized', 'Smooth']
emn, epn = ['Err-', 'Err+']  # difference from mean/median
sln, sdn = ['slope', 'Deviation']
intn = 'Integral'
# Edit `base` if a different location is desired for data.
base = ''
pre = base + 'Data/'

# %% Utilities

def make_spec(spec: dict):
    """
    Return a namedtuple allowing the use of dot notation for dictionaries

    Parameters
    ----------
    spec: dict  Dictionary to be turned into a namedtuple

    Returns
    -------
    namedtuple containing other spec dictionaries
    """
    sType = namedtuple('sType', list(spec))
    return sType(**spec)

def date_string(d):
    """ Return date string from a tuple of (year, month)
    """
    return f'{d[0]}-{d[1]}-01'

def date_index(start, end, freq='MS'):
    """
    Get Pandas DateTime index with the supplied start and end dates

    Parameters
    ----------
    start : tuple(int, int)
        (year, month: 1-index).
    end : tuple(int, int)
        (year, month: 1-index).
    freq : string
        ['MS'|'Y'] : monthly start or yearly (default monthly)

    Returns
    -------
    DateTime index.
    """
    a = date_string(start)
    b = date_string(end)
    return pd.date_range(start=a, end=b, freq=freq)


# %% Functions
def update_modern(f: str):
    """
    Download modern values and save as tab-deliminated file
    """
    spec = dst.specs[f]
    fmt = spec['format']
    headers = {'User-agent': 'Custom User Agent'}
    name = spec['name']
    url = spec['url']
    table = spec.get('table', False)
    origin = spec.get('origin', None)  # used for julian dates
    if table:
        fmt['names'] = [yn] + list(range(1, 13))
        fmt['usecols'] = list(range(13))  # year plus months
    if 'start_year' in spec.keys():
        # File has unwanted info at end of file that could mess
        # up the loading process. Calculate the number of lines
        # that are needed and load just those.
        start_year = spec['start_year']
        start_month = spec.get('start_month', 1)
        now = dt.date.today()
        skip = fmt.get('skiprows', 0)
        if table:
            lines = now.year - start_year + 1 - skip
        else:
            lines = (now.year - start_year - 1) * 12
            lines += 13 - start_month
            lines += now.month - skip  # skipped lines not included
        fmt['nrows'] = lines

    #  new code required to get around lost feature in pandas 3.0
    #  where multiple columns could be combined into a single date
    parse_dates = fmt.pop('parse_dates', False)
    date_format = fmt.pop('date_format', False)
    if parse_dates:
        fmt['index_col'] = False

    response = requests.get(url, headers=headers)
    df = pd.read_csv(StringIO(response.text), **fmt)
    
    if parse_dates:
        # get list of date-like columns
        if isinstance(parse_dates, str):  # single column
            name = parse_dates
            cols = [parse_dates]
            fmt_str = date_format
        else:  # multiple columns
            i = list(parse_dates.keys())[0]
            cols = parse_dates[i]
            # get format for each column
            fmt_str = '-'.join([date_format[col] for col in cols])
        # turn date-like columns into strings and combine
        df[cols] = df[cols].apply(lambda x: x.astype('str'))
        date = df[cols].apply(lambda x: '-'.join(x), axis=1)
        
        df.index = pd.to_datetime(date, format=fmt_str)
        df.index.name = 'Date'  # standard datetime column name
        # remove superfluous columns
        df.drop(columns=cols, inplace=True)
    
    if 'nrows' in spec.keys():  # remove spurious lines
        mx = df.index.argmax()
        df = df.iloc[:mx+1]
    if 'Err5' in df.columns:
        df[emn] = (df[dn] - df['Err5'])/2
        df[epn] = (df['Err95'] - df[dn])/2
        df.drop(columns=['Err5', 'Err95'], inplace=True)
    elif en in df.columns:
        df[emn] = df[en]/2
        df[epn] = df[emn]
        df.drop(columns=[en], inplace=True)
        
    if origin:
        df.index = pd.to_datetime(df.index, origin=origin, unit='D')

    if table:
        # Turn a table with month columns to a long list
        df[yn] = df.index
        df = df.melt(id_vars=[yn], value_name=dn, var_name='Month')
        df['Day'] = 1
        df.index = pd.to_datetime(df[['Year', 'Month', 'Day']])
        df.index.name = 'Date'
        df.drop(columns=['Year', 'Month', 'Day'], inplace=True)
        df.sort_index(inplace=True)
    df.dropna(inplace=True)

    df.label = name
    df.to_csv(pre+spec['file_name'], sep='\t', 
              float_format='%.4f', date_format='%Y-%m-%d')
    return df
  
def update_special(f: str):
    """ Process raw data and save as tab-delimited files
    
        f: str, name of data set
    """
    spec = dst.specs[f]
    name = spec['name']
    if f=='stratvol':
        df = calc_volcano()
    df.label = name
    df.to_csv(pre+spec['file_name'], sep='\t', 
              float_format='%.4f', date_format='%Y-%m-%d')
    return df
    
def load_modern(f: str):
    """ Return files that have been processed as
        tab-delimited files

        f: str, name of data set
    """
    spec = make_spec(dst.specs[f])
    fname = spec.file_name
    df = pd.read_csv(pre+fname, sep='\t', index_col=0, parse_dates=[0])
    if f in dst.specs['temperature']:
        # assume temperatures have a datetime index
        # normalize to Hadcrut data for 1961-90
        df['Raw'] = df[dn].to_numpy()  # force a copy
        df[dn] -= df.loc[(df.index.year>=1961)&(df.index.year<=1990), dn].mean()
        df[dn] += dst.specs['pie_offset']  # pre-industrial era
    if hasattr(df.index, 'month'):
        counts = df[df.columns[0]].groupby(df.index.year).count()
        high_yrs = counts.loc[counts > 24].index.values
        if len(high_yrs) > 2 : # it is daily data
            df['m'] = df.index.strftime('%Y-%m-01')
            df = df.groupby('m').mean()
            df.index = pd.to_datetime(df.index)
    df.spec = ''  # avoid warning for setting columns
    df.spec = spec
    return df

def load_special(f: str):
    """ Return files that have been processed as
        tab-delimited files

        f: str, name of data set
    """
    if f=='stratvol':
        df = load_modern(f)
        return df
    else:
        raise Exception(f'"{f}" not a known data source.')

def update_list():
    """Update modern temperature data files.
    """
    m = 'best,hadcrut'
    m = m.split(',')
    for f in m:
        print(f'loading {f}...')
        update_modern(f)

def combine_co2(df, start=1760):
    # add in years from Antarctic Composite to get longer sequence
    if isinstance(df, pd.Series):
        mf = pd.DataFrame(index=df.index)
        mf[dn] = df.values
        mf[epn] = 0.3598  # Moana Loa error values
        mf[emn] = 0.3598
    else:
        mf = df.copy()
    cf = dst.co2composite()
    win = 15  # a larger window required to ensure sufficient data in early yrs
    fy = mf.index[0]  # first year of modern data
    ly = mf.index[-1]
    cyrs = list(range(start, fy))  # annual ice core years
    myrs = mf.index  # modern years
    cols = [dn, emn, epn]
    temp = pd.DataFrame(index=list(range(start, ly + 1)), columns=cols,
                        dtype=np.float)
    temp.loc[myrs, cols] = mf.loc[myrs, cols]
    mf = mf.combine_first(cf)  # add composite data, index is now float
    # Make composite data into annual
    for yr in cyrs:
        xf = mf.loc[(mf.index >= (yr - win)) & (mf.index <= (yr + win))]
        for col in cols:
            [slope, intercept] = np.polyfit(xf.index, xf[col], 1)
            temp.loc[yr, col] = (yr * slope + intercept)
    return temp 


def load_processed(f: str):
    """
    Load processed tab-delimited data

    Parameters
    ----------
    f: str Name of data

    Returns
    -------
    pandas dataframe
    """
    if f not in dst.specs:
        raise Exception(f'{f} is unknown. Check specs.yaml')
    spec = make_spec(dst.specs[f])
    df = pd.read_csv(pre+spec.file_name, sep='\t', index_col=0, parse_dates=[0],
                     comment='#')
    df.spec = ''  # avoid warning for setting columns
    df.spec = spec
    return df

def get_nino(ix):
    """ Return the requested nino index
    """
    ixs = [12, 3, 34, 4]
    if ix not in ixs:
        raise Exception(f'ix must be one of {ixs}')
    path = f'nino{ix}.long.anom.data.txt'
    src = (f'  Nino {ix} Index:\n' +
           f'    psl.noaa.gov/gcos_wgsp/Timeseries/Nino{ix}/')
    df = pd.read_csv(pre + path, sep='\s+', header=None, skiprows=1,
                     index_col=0, na_values=-99.99)
    df.dropna(inplace=True)
    df.index.name = yn
    nino = df.mean(axis=1)
    nino.index = nino.index.astype(int)
    nino.src = src
    return nino

def calc_volcano(end=None):
    """
    Retrieve aerosol data and turn it into a volcanic aerosol forcing time
    series. Use seasonal solar variability, and albedo at each latitude to
    calculate.

    Parameters
    ----------
    end : tuple(year:int, month:int), or int, optional
        End date or year if it is earlier than data. The default is None.
    annual : bool, default False
        If True, return annual means.

    Returns
    -------
    cvol : pd.Series
        volcanic index (max = 1).

    """
    # Get aerosol data from the GloSSAC satellite instrument data set
    # This is in NetCDF format and requires a free account to access
    # Source: https://asdc.larc.nasa.gov/project/GloSSAC
    # A new version is available at the end of each year.
    url = 'Data/GloSSAC_V2.22.nc'
    gl = xr.open_dataset(url)
    aod = gl.Glossac_Aerosol_Optical_Depth[:,:,2]  # 525 nm
    # time index is integers in the form yyyymm
    time_index = aod.indexes['time']
    start = ((time_index[0] // 100), (time_index[0] % 100))
    last = ((time_index[-1] // 100), (time_index[-1] % 100))
    if not end:
        end = last
    elif hasattr(end, 'month'):
        end = (end.year, end.month)
    dates = date_index(start, end)
    
    # new dataframe with date index
    vol_df = pd.DataFrame(index=dates, columns=aod.lat.values, 
                          dtype=np.float64)
    vol_df.iloc[:aod.values.shape[0],:] = aod.values
    # remove baseline aerosols from 1997 -2005 quiet period
    vol_df -= vol_df[(vol_df.index.year>=1997) & 
                     (vol_df.index.year<=2005)].mean()
    
    deg2rad = np.pi/180.0
    max_tilt_rad = 23.44 * deg2rad  # axial tilt
    # note month is 1-indexed
    tilt_rad = -max_tilt_rad * np.cos((dates.month - 2 + 21/31)/12 * 2 * np.pi)
    lat_rad = vol_df.columns.values * deg2rad
    true_lat_rad = np.add.outer(tilt_rad, lat_rad)
    adjust = np.maximum(0, np.cos(true_lat_rad))  # truncate negative values at 0
    adjust *= np.cos(lat_rad)  # area at polar latitudes less than at equator
    vol_df *= adjust

    # load planet albedo at 0.5° resolution for Dec and Jun, 2022
    # Satellite doesn't have albedo for regions in darkness.
    # Although Sep or Mar would work, more melting has occured for one hemisphere
    # Source: https://neo.gsfc.nasa.gov/view.php?datasetId=MCD43C3_M_BSA&year=2022
    url = 'Data/MCD43C3_M_BSA_2022-06-01_rgb_720x360.SS.CSV'
    albedo = pd.read_csv(url, sep=',', index_col=0, header=0, dtype=np.float64)
    url = 'Data/MCD43C3_M_BSA_2022-12-01_rgb_720x360.SS.CSV'
    south = pd.read_csv(url, sep=',', index_col=0, header=0, dtype=np.float64)
    albedo.loc[albedo.index < 0] = south.loc[south.index < 0]
    
    # Now calculate heating rate for land and ocean
    celsius2kelvin = 274.15
    area_ocean = .71  # ocean area of Earth
    area_land = 1. - area_ocean
    heat_ocean = .95  # amount of incoming heat energy stored in ocean
    heat_land = 1. - heat_ocean
    heat_per_area_ocean = heat_ocean / area_ocean
    heat_per_area_land = heat_land / area_land
    temperature_land = 8.6 + celsius2kelvin  # Berkely Earth
    temperature_global = 14.7 + celsius2kelvin  # Berkely Earth
    temperature_ocean = (temperature_global - temperature_land * area_land) \
                        / area_ocean
    heat_capacity_ocean = heat_per_area_ocean / temperature_ocean
    heat_capacity_land = heat_per_area_land / temperature_land
    ocean_vs_land_capacity = heat_capacity_ocean / heat_capacity_land  # 7.53
    # adjust albedo for ocean
    react = 1 - albedo
    # albedo of 9999 is for ocean or no data
    react[react < 0] = 1. / ocean_vs_land_capacity  # higher capacity less reactive
    react = react.mean(axis=1)
    r_df = pd.DataFrame(react, columns=['React'])
    # combine latitudes to match aerosol data
    r_df['lat'] = np.floor((r_df.index-5)/5)*5 + 7.5
    react = r_df.groupby('lat').mean()
    react = react / react.max()
    vol_df[:] *= react[-77.5:77.5].values.T
    vol = vol_df.mean(axis=1)
    vol.fillna(0, inplace=True)
    return vol

class _DataSource:
    """ Singleton class that interfaces with file data and keeps a copy
        in memory when requested for faster loading.
    """
    # list properties to copy with data in dataframes
    _meta = ['spec']

    def __init__(self):
        self.frames = {}
        with open(base+'specs.yaml', 'r') as file:
            self.specs = yaml.safe_load(file)
        keys = list(self.specs.keys())[2:]
        # add dataframes as methods to get with . notation
        for k in keys:
            self.__dict__.update({k: lambda f=k: self._request(f)})

    def _pull(self, f):
        """ Pull data from data file and return data frame
        """
        if f in self.specs['modern']:
            return load_modern(f)
        elif f in self.specs['special']:
            return load_special(f)
        else:
            return load_processed(f)

    def _request(self, f):
        """ Return a copy of requested data frame given string name
        """
        if f in self.frames:
            return self.frames[f].copy()
        r = self._pull(f)
        # tell dataframe what new properties to copy with .copy()
        r._metadata.extend(self._meta)
        r.index.name = f
        self.frames[f] = r
        return r.copy()

    def reset(self):
        """ Reset the stored dataframes.
        """
        self.frames = {}

dst = _DataSource()