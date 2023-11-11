#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 20:17:30 2020

@author: dan
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tools as tls
import datastore as ds
import datetime as dt
import xarray as xr

dst = ds.dst  # Data Store, an object that loads and stores data in a common format
plt.style.use('clean')

yn, mn, dn, en, sn = ['Year', 'Month', 'Data', 'Error', 'Smooth']

def get_impulse(n, monthly=False):
    """ Return annual (default) or monthly warming curve based on 
        Caldeira and Myhrvold 2013, Using 3 exponent model with mean values
        th0     th1     th2     ta0     ta1     ta2
        .226    .354    .409    .586    7.15    273.7
    """
    # f(t) = 1 - (th0 e^-t/ta0 + th1 e^-t/ta1 + th2 e^t/ta2)
    def f(x, th, ta):
        t = x + 1.  # advance along the curve a bit for current year
        v = (th[0] * np.exp(-t / ta[0]) +
              th[1] * np.exp(-t / ta[1]) +
              th[2] * np.exp(-t / ta[2])).clip(0., 1.)
        return 1. - v

    th = np.array([.238, .345, .416])  # mean values
    ta = np.array([.655, 9.46, 257.1])
    #    th = np.array([.226, .354, .409 ])  # median values
    #    ta = np.array([.586, 7.15, 273.7])
    if monthly:
        p = 1/12
    else:
        p = 1
    x = np.arange(0, n*p, p)
    impulse = pd.DataFrame(index=x)
    impulse[dn] = f(x, th, ta)
    return impulse

def convolve_impulse(data: pd.Series, 
                     impulse: pd.DataFrame=None, 
                     monthly: bool=False):
    """ Convolve data with forcing impulse.
        Assumes impulse has +/- values as well.

        data: Series
        impulse: DataFrame
        monthly: boolean  True if monthly data present, default False
    """
    if impulse is None:
        impulse = get_impulse(len(data), monthly=monthly)
    n = len(data)
    kernel = np.zeros(2 * n - 1)
    change = data.copy()
    change.iloc[1:] -= data.iloc[:-1].values
    change.iloc[0] = 0.0
    result = pd.DataFrame(index=data.index)
    kernel[-n:] = impulse[dn].values
    c = np.convolve(change.values, kernel, 'valid')
    result[dn] = c[0:n]
    return result

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

def calc_volcano(end=None):
    """
    Retrieve aerosol data and turn it into a volcanic aerosol forcing time
    series. Use seasonal solar variability, and albedo at each latitude to
    calculate.

    Parameters
    ----------
    end : tuple(int, int), optional
        End date if it is later than data. The default is None.

    Returns
    -------
    cvol : pd.Series
        volcanic index (max = 1).

    """
    # Get aerosol data from the GloSSAC satellite instrument data set
    # This is in NetCDF format and requires a free account to access
    # Source: https://asdc.larc.nasa.gov/project/GloSSAC
    url = 'Data/GloSSAC_V2.21.nc'
    gl = xr.open_dataset(url)
    aod = gl.Glossac_Aerosol_Optical_Depth[:,:,2]  # 525 nm
    # time index is integers in the form yyyymm
    time_index = aod.indexes['time']
    start = ((time_index[0] // 100), (time_index[0] % 100))
    last = ((time_index[-1] // 100), (time_index[-1] % 100))
    if not end:
        end = last
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
    # add heating curve to volcanic values by convolution
    cvol = convolve_impulse(vol, monthly=True)
    return cvol

def compile_vars(source='hadcrut', index='N34'):
    temp = ds.load_modern(source, annual=False)
    start = '1980-01-01'
    end = temp.index[-1]
    temp = temp.loc[start:end]
    df = pd.DataFrame(index=temp.index)
    df['vol'] = calc_volcano((end.year, end.month))
    df['enso'] = dst.enso()[index].loc[start:end]
    df['pdo'] = dst.pdo().loc[start:end]
    solar = dst.solar().loc[start:end]
    csolar = convolve_impulse(solar, monthly=True)
    df['solar'] = csolar
    
    # detrend temperature
    xi = np.arange(len(temp))
    y = temp.values
    slope, intercept = np.polyfit(xi, y, 1)
    dy = y - intercept - slope * xi
    dtemp = pd.Series(index=temp.index, data=dy, dtype=np.float64)
    sigma = dtemp.std()
    

def plotTempTrend(index=None):
    """ Plot the monthly temperature trend to 2070
    
        index: string, if provided, remove ENSO signal. Possible values
               are: [N12, N3, N4, N34] corresponding to NINO index.
    """
    def get_date(y):
        xpi = int((y - intercept) / slope)
        return xp[xpi]
        
   # Do analysis
    
    source = 'hadcrut'
    df = ds.load_modern(source, annual=False)
    spec = df.spec  # specs that were added by DataStore module
    start = '1980-01-01'
    end = '2070-01-01'
    df = df.loc[df.index>=start]
    x = df.index.values
    xi = np.arange(len(x))  # Index
    xp = pd.date_range(start, end, freq='MS', inclusive='left')  # projection
    xpi = np.arange(len(xp))  # projection index
    y = df.Data.values
    
    if index:
        # remove ENSO signal
        slope, intercept = np.polyfit(xi, y, 1)  # get LSF trend
        dy = y - (slope * xi + intercept)  # detrend the data
        enso = dst.enso()[index]
        i = len(x) - len(enso)  # Align start dates, ENSO data starts later
        x = x[i:]
        xi = xi[i:]
        dy = dy[i:]
        scale = dy.max()/enso.max()
        dy -= enso.values * scale
        y = dy + slope * xi + intercept
        enso_text = f'(NINO{index[1:]} signal removed)'
    
    slope, intercept = np.polyfit(xi, y, 1)  # get LSF trend
    ys = slope * xi + intercept
    ysp = slope * xpi + intercept
    dy = y - ys  # detrend the data
    sigma = dy.std()

    dates = {1.5:get_date(1.5), 
             2.0:get_date(2.0)}

    fig = plt.figure('Projection')
    fig.clear()  # May have been used before
    ax = fig.add_subplot(111)

    ax.plot(x, y, 'k+', alpha=0.3)     # data
    ax.plot(xp, ysp, 'b-', lw=1) # trend
    ax.fill_between(xp, ysp+2*sigma, ysp-2*sigma, color='b', alpha=.12)
    ax.fill_between(xp, ysp+sigma, ysp-sigma, color='b', alpha=.12)
    
    ymin = y.min()
    xmin = x[0]
    
    enso_text = ''
    if not index:
        ax.text(x[-1], y[-1], f'  {df.index[-1]:%b %y}', 
                ha='left', va='center', size='small')  # last month
        lastx = df.loc[df.index.year<2020, 'Data'].idxmax()
        lasty = df.loc[lastx, 'Data']
        ax.text(lastx, lasty, f'{lastx:%b %y}  ', 
                ha='right', va='center', size='small')  # last month

        
    for k in dates.keys():
        ax.hlines(k, xmin, dates[k], color='k', lw=0.5, ls=':')
        ax.vlines(dates[k], ymin, k, color='k', lw=0.5, ls=':')
        ax.text(dates[k], k, dates[k].year, ha='left', va='top', weight='bold')

    ax.text(xp[-1], ysp[-1]+sigma*2, '95% Range', va='center')
    ax.text(xp[-1], ysp[-1]+sigma, '68% Range', va='center')
    ax.text(xp[24], 2.2, "Note: This is a very simplistic projection based "+ \
            "only on past trends", size='large')

    tls.byline(ax)
    tls.titles(ax, f"Temperature Projection to 2070 {enso_text}",
               f"{spec.name} monthly change from pre-industrial (°C)")
    plt.show()
    return


def plotTempVar():
    """ Plots variability of global temperature with and without
        ENSO removed.
    """
    def new_axes(name, title, ylabel):
        fig = plt.figure(name)
        fig.clear()  
        ax = fig.add_subplot(111)  
        tls.byline(ax)
        tls.titles(ax, title, ylabel)
        return ax
    
    def plotOne(ax, data, sigma, years=None, labels=None):
        """ Plot one axes given a Pandas Series as data
        """
        if labels:
            years = None
        ax.plot(data.index, data.values, 'k+', alpha=.3)
        ax.axhline(color='k', lw=.5)
        for i in [1, 2]:
            ax.fill_between(data.index, i*sigma, -i*sigma, color='b', alpha=.12)
        labels = labelYears(ax, data, sigma, years, labels=labels)
        return labels
    
    def max_years(data, years=None):
        if not years:
            years = [1990, 1998, 2016, 2022]
        labels = []
        for yr in years:
            start = dt.datetime(yr-2,1,1)
            end = dt.datetime(yr+2,1,1)
            date_range = data.loc[start:end]
            labels.append(date_range.idxmax())
        return labels
            
    def labelYears(ax, data, sigma, years=None, labels=None):
        """ Label the warmest month in a range centred around the supplied
            years. 
            
            data: Pandas series
            sigma: float
            years: list of ints, years to search for max temperatures
            labels: list of Timestamp values to use with data.loc[], or None
            
            returns: labels
        """
        if not labels:
            labels = max_years(data, years)
        for x in labels:
            y = data.loc[x]
            ys = y/sigma
            t = f"{x:%b %y}\n{ys:.1f}σ"
            ax.text(x, y, t, ha='center', 
                    va='bottom', size='small')
        return labels
    
    source = 'hadcrut'
    index = 'N34'
    df = ds.load_modern(source, annual=False)
    spec = df.spec  # specs that were added by DataStore module
    enso = dst.enso()
    mei = ds.load_modern('mei', annual=False)
    mei = mei.loc[enso.index[0]:]
    enso['_MEI'] = mei
    enso = enso[index]
    start = enso.index[0]
    data = df.loc[df.index>=start, 'Data']
    xi = np.arange(len(data))  # Index
    slope, intercept = np.polyfit(xi, data.values, 1)
    data -= slope * xi + intercept  # detrend data
    sigma = data.std()
    labels = max_years(data)
    
    # Plot temperature and Nino Index
    ax = new_axes('Deviation',
                  "Temperature Deviation from Trend",
                  f"{spec.name} Monthly Global Temperature (ºC)")
    plotOne(ax, data, sigma)
    # plot Nino index over top
    scale = data.max()/enso.max()
    y = enso.values * scale
    ax.plot(enso.index, y, 'r-', lw=.75)
    ax.text(enso.index[-1], y[-1], f"Niño {index[1:]} Index\n  x {scale:.3f}",
            color='r', size='small')
    
    # Same plot with temperature averaged over 3 months
    adata = data.rolling(window=3, min_periods=1, center=True,
                         closed='both').mean()
    alabels = max_years(adata)
    ax = new_axes('Averaged',
                  "Averaged, Detrended Temperature",
                  f"{spec.name} Monthly Global Temperature (ºC) (3-month average)")
    asigma = adata.std()
    plotOne(ax, adata, asigma, labels=alabels)
    scale = adata.max()/enso.max()
    y = enso.values * scale
    ax.plot(enso.index, y, 'r-', lw=.75)
    ax.text(enso.index[-1], y[-1], f"Niño {index[1:]} Index\n  x {scale:.3f}",
            color='r', size='small')
    
    # Plot with N3 removed from temperature
    ax = new_axes('Less_ENSO',
                  "Temperature Deviation from Trend with Niño Index 3 Removed",
                  f"{spec.name} Monthly Global Temperature (ºC)")
    edata = data - enso * scale
    plotOne(ax, edata, sigma, labels=labels)
    
    # Plot with 3-month rolling average applied to temperature
    ax = new_axes('Averaged_Less_ENSO',
                  f"Averaged, Detrended Temperature with Niño {index[1:]} Index Removed",
                  f"{spec.name} Monthly Global Temperature (ºC) (3-month average)")
    scale = adata.max()/enso.max() * .8
    print(f"scale: {scale}")
    edata = adata - enso * scale
    plotOne(ax, edata, asigma, labels=alabels)
    
    plt.show()
    return

plotTempVar()

def test_enso():
    """ Find the ENSO index that best reduces the temperature variation.
    """
    
    def compare(data, test):
        """ Compare ENSO with supplied data and test function
        """
        enso = dst.enso()
        enso = enso.loc[data.index]  # use comparable data points
        bar = test(data.values)  # the bar to get under
        min_bar = bar
        min_index = ''
        for c in enso.columns:
            # Set up matrix for least squares fit equation
            ei = enso[c].values
            #ei *= np.abs(ei)**.1  # trying different powers
            A = np.vstack([ei, np.ones(n)]).T
            # Get best fit
            rc = np.linalg.lstsq(A, data.values, rcond=None)[0]
            ry = data.values - rc[0] * ei
            result = test(ry)
            print(f"{c}: {rc[0]:.5f} - {result:.5f}, {result/bar*100:.1f}%")
            if result < min_bar:
                min_bar = result
                min_index = c
        return min_index
    
    source = 'hadcrut'
    enso = dst.enso()
    temp = ds.load_modern(source, annual=False)
    temp = temp.loc[temp.index>=enso.index[0], 'Data']  # The index starts after temp data
    
    # Remove the temperature trend
    n = len(temp)
    ix = np.arange(n)
    slope, intercept = np.polyfit(ix, temp.values, 1)
    temp.values -= (slope * ix + intercept)

    # Test by reducing the std dev
    print("\nReducing Std Dev\n")   
    compare(temp, np.std)
    
    #  apply 3-month moving average
    temp = temp.rolling(window=3, min_periods=1, center=True,
                                  closed='both').mean()
    print("\nReducing with 3-month average\n")
    compare(temp, np.std)
    