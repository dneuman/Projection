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

# %% Data Tools

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

def calc_volcano(end=None, annual=False):
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
    url = 'Data/GloSSAC_V2.21.nc'
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
    elif annual:
        end = (end, 12)
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
    
    if annual:
        vol = vol.groupby(vol.index.year).mean()
    return vol

# %% Data Processing

def compile_vars(source='hadcrut', annual=False):
    """ Return DataFrame containing temperature, trend line, detrended 
        temperature, and environmental factors that might affect temperature.
    """
    temp = ds.load_modern(source, annual=annual)    
    start = pd.to_datetime('1980-01-01')
    if annual:
        start = start.year
    end = temp.index[-1]
    df = pd.DataFrame()
    df['temp'] = temp.loc[start:end, 'Data']
    df.spec = ''  # prevent warning about adding columns this way
    df.spec = temp.spec  # add file specifications
    # get trend line
    xi = np.arange(len(df))
    y = df.temp.values
    slope, intercept = np.polyfit(xi, y, 1)
    df['trend'] = slope * xi + intercept
    df['detrend'] = (df.temp - df.trend)
    vol = calc_volcano(end, annual=annual)
    # apply ocean warming curve to volcanic forcing
    df['vol'] = convolve_impulse(vol, monthly=(not annual))
    enso = dst.enso(annual=annual).loc[start:end]
    df[enso.columns] = enso
    # # The Pacific Decadal Oscillation does not appear to have an effect
    # df['pdo'] = dst.pdo(annual=False).loc[start:end]
    solar = dst.solar(annual=annual).loc[start:end] 
    # Remove any longterm trend. This will already be removed from temperature.
    y = solar.values
    slope, intercept = np.polyfit(xi, y, 1)
    yt = slope * xi + intercept
    solar.Data -= yt  # solar is a single-column dataframe
    df['solar'] = convolve_impulse(solar.Data, monthly=True)
    df.fillna(0, inplace=True)
    return df

def fit_vars(source='hadcrut', annual=False, df=None):
    if df is None:
        df = compile_vars(source, annual=annual)
    sigma = df.detrend.std()
    print(f'Original standard deviation was: {sigma:.4f}°C')
    df['linear'] = np.arange(len(df))  # true linear trend
    cols = df.columns[3:]
    A = np.vstack([df[cols].to_numpy().T,
                   np.ones(len(df))]).T
    # The last column of A is constant, for mx + b
    c = np.linalg.lstsq(A, df.temp.values, rcond=None)[0]
    offset = c[-1]
    df[cols] *= c[:-1]  # this does not include the constant offset c[-1]
    df['vars'] = df[cols[:-1]].sum(axis=1)  # all variables except trend
    vars_offset = df.vars.mean()
    df['vars'] -= vars_offset
    df['linear'] += (offset + vars_offset)  # mx + b
    df['reduced'] = df.temp - df.linear - df.vars
    nsigma = df.reduced.std()
    print(f'New standard deviation is: {nsigma:.4f}°C')
    print(f'Reduction of {(sigma-nsigma)/sigma*100:.1f}%')
    rate = 10 * ((not annual) * 11 + 1)
    print(f'New slope is {(c[-2]*rate):.3f}°C/decade')
    r2 = tls.R2(df.detrend.values, df.vars.values)
    print(f'R² value is {r2:.3f}')
    return df    

# %% Plotting helpers


def new_axes(name, title, ylabel):
    fig, ax = plt.subplots(1, 1, num=name, clear=True)
    tls.byline(ax)
    tls.titles(ax, title, ylabel)
    return ax

def new_fig_rows(name, title, ylabel, num=1):
    fig, axs = plt.subplots(num, 1, num=name, clear=True, sharex=True)
    fig.subplots_adjust(hspace=0)
    tls.byline(axs[-1])
    tls.titles(axs[0], title, ylabel)
    return axs

def plot_one(ax, data, sigma, years=None, labels=None):
    """ Plot one axes given a Pandas Series as data
    """
    if labels:
        years = None
    ax.plot(data.index, data.values, 'k+', alpha=.3)
    ax.axhline(color='k', lw=.5)
    for i in [1, 2]:
        ax.fill_between(data.index, i*sigma, -i*sigma, color='b', alpha=.12)
    labels = label_years(ax, data, sigma, years, labels=labels)
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
        
def label_years(ax, data, sigma, years=None, labels=None):
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
    
# %% Plotting Functions

def plotTempTrend(source='hadcrut', annual=False):
    """ Plot the monthly temperature trend to 2080
    """
    def get_date(y):
        xpi = int((y - intercept) / slope)
        if xpi > len(xp):
            return xp[-1]
        return xp[xpi]
        
    # Do analysis
    df = fit_vars(source, annual)
    # Important columns:
        # temp: temperature
        # trend: trend calculated from the temperature
        # detrend: temperature - trend
        # vars: natural variation fit to the detrend values
        # reduced: temp - vars
        # linear: real trend from reduced values
    
    start = df.index[0]
    if annual:
        end = 2080
        xp = np.arange(start, end+1)
    else:
        end = '2080-01-01'
        xp = pd.date_range(start, end, freq='MS', inclusive='left')  # projection
    x = df.index.values
    xpi = np.arange(len(xp))  # projection index
    
    def plot(y, yt, ytp, win_name='projection', title_app=''):
        """ Plot values against trend and variance
        
            y: values
            yt: trend
            ytp: projected trend
        """
        figname = win_name
            
        sigma = (y - yt).std()
    
        dates = {1.5:get_date(1.5), 
                 2.0:get_date(2.0)}
        ymin = 0
        xmin = x[0]
        slope = (yt[-1] - yt[0]) / len(yt)
        lh = []  # legend handle
        fig = plt.figure(figname, clear=True)
        ax = fig.add_subplot(111)
        ax.set_ylim(ymin, 2.5)
#TODO finish legend
        lh = lh.append(ax.plot(x, y, 'k+', alpha=0.5, label='')) # data
        ax.plot(xp, ytp, 'b-', lw=1) # trend
        ax.fill_between(xp, ytp+2*sigma, ytp-2*sigma, color='b', alpha=.12)
        ax.fill_between(xp, ytp+sigma, ytp-sigma, color='b', alpha=.12)
        for k in dates.keys():
            if annual:
                year = dates[k]
            else:
                year = dates[k].year
            ax.hlines(k, xmin, dates[k], color='k', lw=0.5, ls=':')
            ax.vlines(dates[k], ymin, k, color='k', lw=0.5, ls=':')
            ax.text(dates[k], k, year, ha='right', va='bottom', weight='bold')

        ax.text(xp[-1], ytp[-1]+sigma*2, '95% Range', va='bottom')
        ax.text(xp[-1], ytp[-1]+sigma, '68% Range', va='center')
        ax.text(xp[-1], ytp[-1], f'σ = {sigma:.3f}°C', va='top')
        text = ("Note: This is a very simplistic projection based only on past trends\n"+
                "Natural Influences are El Niño, volcanic activity, and solar.")
        ax.text(1982, 2.2, text, size='large')
        rate = 10 * ((not annual) * 11 + 1)  # 10 or 120
        change = 'monthly'
        if annual:
            change = 'annual'
        ax.text(get_date(1.75), 1.75, f"{slope*rate:.3f}°C/decade", va='top')
        tls.titles(ax, f"Temperature Projection to 2080{title_app}",
                   f"{df.spec.name} {change} change from pre-industrial (°C)")
        tls.byline(ax)
        plt.show()
        
        return ax
    
    handle = []  # 
 
    # #=== Plot Observed Trend ===

    # y = df.temp.values
    # yt = df.trend.values
    # slope = (yt[-1] - yt[0]) / len(yt)
    # intercept = yt[0]
    # ytp = slope * xpi + intercept
    # ax = plot(y, yt, ytp)
    
    #=== Plot trend compared with natural influences ===
    
    y = df.temp.values
    yt = df.trend.values
    slope = (yt[-1] - yt[0]) / len(yt)
    intercept = yt[0]
    ytp = slope * xpi + intercept
    ax = plot(y, yt, ytp, win_name='projection_compare',
              title_app=', Comparing Natural Influences')
    y = (df.vars + df.linear).values
    ax.plot(df.index.values, y)
    
    #=== Plot Trend with natural influences removed ===
    
    y = df.reduced.values + df.linear.values
    yt = df.linear.values
    slope = (yt[-1] - yt[0]) / len(yt)
    intercept = yt[0]
    ytp = slope * xpi + intercept
    ax = plot(y, yt, ytp, win_name='projection_reduced',
              title_app=', Natural Influences Removed')
    
    plt.show()
    return

def plotTempVar(source='hadcrut'):
    """ Plots variability of global temperature with and without
        external variation (ENSO, volcanic sulphates, and solar) removed.
    """
    df = compile_vars(source)
    df = fit_vars(df)
    labels = max_years(df.detrend)
    
    # Plot temperature and Nino Index
    axs = new_fig_rows('Deviation',
                       "Temperature Deviation from Trend",
                       f"{source.capitalize()} Monthly Global Temperature (ºC)",
                  num=2)
    sigma = df.detrend.std()
    ax = axs[0]
    plot_one(ax, df.detrend, sigma, labels=labels)
    # plot natural influences over top
    ax.plot(df.index, df.vars.values, 'r-', lw=1)
    ax.text(df.index[-1], df.vars[-1], " Natural\n Influences",
            color='r', size='small', va='center', weight='bold')
    y1 = df.trend[0]
    y2 = df.trend[-1]
    dx = len(df) / 120
    slope = (y2 - y1)/dx
    ax.text(df.index[-1], -.01, f' Trend:\n {slope:.4f}°C/decade',
            size='small', va='top')
    ylim = ax.get_ylim()
    text = 'Natural influences are NINO Indexes, Volcanic sulphates, Solar, and PDO'
    ax.text(.99, 0.01, text, ha='right', transform=ax.transAxes,
            color='r', size='small')
    ax = axs[1]
    nsigma = df.reduced.std()
    plot_one(ax, df.reduced, nsigma, labels=labels)
    ax.set_ylim(ylim)
    ax.text(df.index[0], 2.1*nsigma, "Natural Influences Removed",
            color='k', weight='bold')
    y1 = df.real[0]
    y2 = df.real[-1]
    slope = (y2 - y1)/dx
    ax.text(df.index[-1], -.01, f' Trend:\n {slope:.4f}°C/decade',
            size='small', va='top')
   
    plt.show()
    return

def plotInfluences(df=None):
    """
    Show how the fitted natural influences compare

    Parameters
    ----------
    df : pd.DataFrame, optional
        Dataframe of natural variances. The default is None.

    Returns
    -------
    None.

    """
    if df is None:
        df = fit_vars()
    e_cols = 'N12 N3 N4 N34'.split()
    cols = 'enso vol solar pdo'.split()
    name = 'All ENSO Volcanic Solar PDO'.split()
    df['enso'] = df[e_cols].sum(axis='columns')
    # plot major influences
    axs = new_fig_rows('Influences', 
                       'Natural Influences', 
                       'Temperature Effect °C', num=len(cols)+1)
    axs[0].figure.set_size_inches(9,9)
    for c, i in zip(['vars']+cols, range(len(cols)+1)):
        axs[i].plot(df[c])
        axs[i].set_ylim((-.25, .25))
        axs[i].text(df.index[0], -.1, name[i], weight='bold')
    # Break down the ENSO index
    axs = new_fig_rows('ENSO',
                       'ENSO Components',
                       'Temperature Effect °C', num=len(e_cols)+1)
    axs[0].figure.set_size_inches(9,9)
    names = ['Combined'] + e_cols
    for c, i in zip(['enso']+e_cols, range(len(cols)+1)):
        axs[i].plot(df[c])
        axs[i].set_ylim((-.25, .25))
        axs[i].text(df.index[0], -.1, names[i], weight='bold')

    plt.show()
    
def plotHist(df=None, num=3):
    """ Plot histograms of detrended temperature and with natural influences
        removed.
        
        num: number of bins per standard deviation
    """
    def normal(x, mu, sigma):
        y = 1/(sigma * np.sqrt(2 * np.pi)) \
            * np.exp( - (x - mu)**2 / (2 * sigma**2))
        return y
    
    if df is None:
        df = fit_vars()
    cols = ['detrend', 'reduced']
    titles = {cols[0]:'Global Temperature',
              cols[1]:'Natural Influences Removed'}
    fig, axs = plt.subplot_mosaic([cols], num='hist', clear=True, 
                                  sharey=True, layout='tight')
    fig.suptitle('Histograms of Monthly Temperatures',
                 ha='left', x=0.1)
    fig.supxlabel('Deviation from Trend (°C)')
    axs[cols[0]].set_ylabel('Number of Months')
    fig.subplots_adjust(wspace=0.02)
    size = {}
    std = {}
    bcols = {}
    pdf = pd.DataFrame(index=np.arange(-4*num, 4*num+1))
    d = df[cols].copy()
    for c in cols:
        axs[c].set_title(titles[c], loc='left')
        std[c] = d[c].std()
        size[c] = std[c] / num  # bin sizes are relative to std dev
        bcols[c] = 'bins_' + c
        pcols = [c, bcols[c]]
        d[bcols[c]] = d[c] // size[c]  # put each data point in a bin
        pdf[c] = d[pcols].groupby(bcols[c]).count()
        edges = np.arange(pdf.index[0], pdf.index[-1]+2) * size[c]
        axs[c].stairs(pdf[c].values, edges, fill=True)
        x = edges[:-1]+size[c]/2
        axs[c].plot(x, size[c]*len(d)*normal(x, 0, std[c]),
                lw=2)
        if c == cols[0]:
            xlim = axs[c].get_xlim()
        else:
            axs[c].set_xlim(xlim)
    plt.show()
        
def plotOceanWarming():
    """ Plot ocean warming curve demonstration
    """
    df = pd.DataFrame(index=np.arange(200))
    cols = ['Warming', 'Cooling', 'Vol']
    wn, cn, vn = cols
    rn = ' Result'
    df[wn] = 0.
    # Add a heating step function
    df.loc[df.index > 25, wn] = 1.
    # Apply the ocean warming result
    # convolve_impulse converts annual values to a sum of steps
    df[wn+rn] = convolve_impulse(df[wn])
    # Add a cooling step function
    df[cn] = df[wn]
    df.loc[df.index > 75, cn] = -1.
    df.loc[df.index > 125, cn] = 0.
    df[cn+rn] = convolve_impulse(df[cn])
    # 
    vol = calc_volcano()
    # normalize
    vol /= -vol.max()
    start = 124
    df[vn] = vol.iloc[start:(start+200)].values
    #df[vn] = vol.loc[vol.index.year > 1988].head(200).values
    df[vn+rn] = convolve_impulse(df[vn], monthly=True)
    # Plot curves
    fig, axs = plt.subplots(3, sharex=True, num='Warming', 
                            clear=True)
    fig.suptitle('Illustration of Ocean Warming',
                 ha='left', x=0.1)
    axs[-1].set_xlabel('Years (Months for Eruption)')
    titles = ['A) Ocean Warming Curve',
              'B) Ocean Warming and Cooling',
              'C) Cooling from 1991 Mount Pinatubo Eruption (months)']
    for ax, col, title in zip(axs, cols, titles):
        ax.plot(df[col])
        ax.plot(df[col+rn])
        ax.set_title(title, loc='left', weight='bold')
    adj = 0.02  # text positioning adjustment
    axs[0].text(100, df[wn][100]-adj, 'Temperature Influence', 
                color='C0', va='top')  
    axs[0].text(100, df[wn+rn][100]-adj, 'Resulting Temperature',
                color='C1', va='top')
    
    axs[0].annotate(f'{df[wn+rn][50]*100:.0f}% warming\nafter 25 years',
                    (50, df[wn+rn][50]), xytext=(20, -20), 
                    textcoords='offset points', va='top',
                    arrowprops=dict(width=2, headwidth=7, headlength=5))
    plt.show()
    
    
    