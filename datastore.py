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

#%precision 2
yn, mn, dn, en, nn, sn = ['Year', 'Month', 'Data',
                          'Error', 'Normalized', 'Smooth']
emn, epn = ['Err-', 'Err+']  # difference from mean/median
sln, sdn = ['slope', 'Deviation']
intn = 'Integral'
# Edit `base` if a different location is desired for data.
base = ''
pre = base + 'Data/'


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
        spec['names'] = list(range(1, 13))
    if 'start_year' in spec.keys():
        # File has unwanted info at end of file that could mess
        # up the loading process. Just calculate the number of lines
        # that are needed and load just those.
        start_year = spec['start_year']
        start_month = spec.get('start_month', 1)
        now = dt.date.today()
        if table:
            lines = now.year - start_year + 1
        else:
            lines = (now.year - start_year - 1) * 12
            lines += 13 - start_month
            lines += now.month - 2  # skipped lines not included
        spec['nrows'] = lines
    response = requests.get(url, headers=headers)
    df = pd.read_csv(StringIO(response.text), **fmt)
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
        # force column names to numbers
        new_names = dict(zip(df.columns, list(range(1, 13))))
        df.rename(columns=new_names, inplace=True)
        df = df.melt(id_vars=[yn], value_name=dn, var_name='Month')
        df['Day'] = 1
        df.index = pd.to_datetime(df[['Year', 'Month', 'Day']])
        df.index.name = 'Date'
        df.drop(columns=['Year', 'Month', 'Day'], inplace=True)
        df.sort_index(inplace=True)
    df.dropna(inplace=True)

    df.to_csv(pre+spec['save_as'], sep='\t', 
              float_format='%.4f', date_format='%Y-%m-%d')
    df.label = name
    return df
  
def load_modern(f: str, annual=True):
    """ Return files that have been processed as
        tab-delimited files

        f: str, name of data set
        annual: bool, return annual data, default True
    """
    spec = make_spec(dst.specs[f])
    fname = spec.save_as
    df = pd.read_csv(pre+fname, sep='\t', index_col=0, parse_dates=[0])
    if f in dst.specs['temperature']:
        # assume temperatures have a datetime index
        # normalize to Hadcrut data for 1961-90
        df[dn] -= df.loc[(df.index.year>=1961)&(df.index.year<=1990), dn].mean()
        df[dn] += dst.specs['pie_offset']  # pre-industrial era
    if hasattr(df.index, 'month'):
        counts = df[df.columns[0]].groupby(df.index.year).count()
        if annual:
            low_yrs = counts.loc[counts < 12].index.values
            if len(low_yrs) < 3:
                # if there are more than 2 low_counts, it means that
                # only annual data is given.
                for y in low_yrs:
                    i = df.loc[df.index.year==y].index
                    df.drop(index=i, inplace=True)
                df = df.groupby(df.index.year).mean()
        else: # return monthly data
            high_yrs = counts.loc[counts > 24].index.values
            if len(high_yrs) > 2 : # it is daily data
                df['m'] = df.index.strftime('%Y-%m-01')
                df = df.groupby('m').mean()
                df.index = pd.to_datetime(df.index)
    df.spec = ''  # prevent error message about creating columns this way
    df.spec = spec
    return df

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
    df = pd.read_csv(pre+spec.save_as, sep='\t', index_col=0, parse_dates=[0],
                     comment='#')
    df.spec = ''
    df.spec = spec
    return df

def load_special(f: str):
    """
    Load data requiring special processing
    Parameters
    ----------
    f: str Name of data

    Returns
    -------
    pandas dataframe
    """
    raise Exception(f'"{f}" not a known data source.')

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
            self.__dict__.update({k: lambda f=k, annual=True: self._request(f, annual)})

    def _pull(self, f, annual=False):
        """ Pull data from data file and return data frame
        """
        if f in self.specs['modern']:
            return load_modern(f, annual)
        elif f in self.specs['special']:
            return load_special(f)
        else:
            return load_processed(f)

    def _request(self, f, annual=False):
        """ Return a copy of requested data frame given string name
        """
        if f in self.frames:
            has_month = hasattr(self.frames[f].index[0], 'month')
            if annual == (not has_month):
                return self.frames[f].copy()
        r = self._pull(f, annual)
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