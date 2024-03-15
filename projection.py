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
import Trend as tr
import datetime as dt
import xarray as xr
import scipy.stats as stats

dst = ds.dst  # Data Store, an object that loads and stores data in a common format
plt.style.use('clean')
pd.options.display.float_format = '{:.4f}'.format  # change print format
pd.options.display.width = 70
np.set_printoptions(precision=5, linewidth=70)

yn, mn, dn, en, sn = ['Year', 'Month', 'Data', 'Error', 'Smooth']

# %% Data Tools

def get_impulse(n: int, annual: bool=True):
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
    if annual:
        p = 1
    else:
        p = 1/12
    x = np.arange(0, n*p, p)
    impulse = pd.DataFrame(index=x)
    impulse[dn] = f(x, th, ta)
    return impulse

def get_simple_impulse(n: int, r: float)->float:
    ''' Return a function that approaches 1 over time.
    
        r: (float) Rate of growth. Larger is slower.
    '''
    x = np.arange(1, n+1)
    df = pd.DataFrame(index=(x-1))
    df[dn] = 1. - np.exp(-x/r)
    return df

def convolve_impulse(data: pd.Series, 
                     impulse: pd.DataFrame=None, 
                     annual: bool=True):
    """ Convolve data with forcing impulse.
        Assumes impulse has +/- values as well.

        data: Series
        impulse: DataFrame
        monthly: boolean  True if monthly data present, default False
    """
    if impulse is None:
        impulse = get_impulse(len(data), annual=annual)
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

def fit(data, vars):
    ''' Fit a set of variables to data
    
        data and vars must have the same number of rows. Returns the scaled
        vars data, with a column containing a constant.
    '''
    n = len(data)
    A = np.hstack([vars.to_numpy(), np.ones((n, 1))])
    c = np.linalg.lstsq(A, data.to_numpy(), rcond=None)
    return c[0]

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

def chow(res, res1, res2, k):
    ''' Return the result of the Chow f-statistic test to determine
        the likelihood that the residuals of a model of two sets
        of data can be better explained by the model of the combined
        data.
        
        res: (array-like) Residuals of the combined data
        
        res1, res2: (array-like) Residuals of the two portions of
                    data each modeled separately.
                    
        k: (int) Number of parameters in the model
    '''
    Dn = k
    Dd = len(res1) + len(res2) - 2*k
    S = (res * res).sum()
    S12 = (res1 * res1).sum() + (res2 * res2).sum()
    C = ((S - S12)/Dn) / (S12 / Dd)
    p = stats.f.sf(C, Dn, Dd)  # Dd is always larger
    r = pd.Series([C, p, Dn, Dd], 
                  index=['Chow', 'p', 'DoF1', 'DoF2'])
    return r

# %% Data Processing

def compile_vars(source='hadcrut'):
    """ Return DataFrame containing temperature, trend line, detrended 
        temperature, and environmental factors that might affect temperature.
    """
    temp = ds.load_modern(source, annual=False)    
    start = pd.to_datetime('1980-01-01')
    end = temp.index[-1]
    df = pd.DataFrame(index=temp.loc[start:end].index)
    df.spec = ''  # avoid warning for setting columns
    df.spec = temp.spec
    df['temp'] = temp.loc[start:end, dn]
    df['lowess'] = tr.lowess(df.temp, pts=15*12)  # this is the local trend, not a line
    df['flat'] = df.temp - df.lowess
 
    df['vol'] = calc_volcano(end, annual=False)
    df['solar'] = dst.solar(annual=False).loc[start:end]
    df.solar -= df.solar.mean()  # remove offset
    enso = dst.enso(annual=False).loc[start:end]
    df[enso.columns] = enso
        
    df.fillna(0, inplace=True)
    return df

def adjust_vars(df, adj, val):
    ''' Adjust the supplied columns using the requested adjustment
        method. Return the adjusted columns
        
        df: (DataFrame) Contains ONLY columns to be adjusted
        
        cols: (iterable|str) Names of the columns to adjust. Can be one
              column name
        
        adj: (str) ['cmip'|'exp'|'lag'] Adjustment type for each 
             column
        
        val: (float or int) Value to use for each column
    '''
    n = len(df)
    cols = df.columns
    vars = pd.DataFrame(index=df.index, columns=cols)
    vars[cols] = 0.0
    for i, col in enumerate(cols):
        if adj == 'cmip':
            vars[col] = convolve_impulse(df[col], annual=False)
        elif adj == 'exp':
            if val < 0.:  # no change, equivalent to doing nothing
                vars[col] = df[col].to_numpy()  # simple copy
            elif val == 0.:  # use the cmip model
                vars[col] = convolve_impulse(df[col], annual=False)
            else:
                impulse = get_simple_impulse(n, val)
                vars[col] = convolve_impulse(df[col], impulse, annual=True)
        elif adj == 'lag':
            val = int(val)
            vars.iloc[val:, i] = df[col].iloc[:n-val]
        else:  # no change
            vars[col] = df[col]
    vars.fillna(0, inplace=True)
    return vars  

def adjust_combined(df, adjs):
    cols = dict(vol=['vol'], solar=['solar'], 
                enso='N12 N3 N4 N34'.split())
    
    for col in cols.keys():
        c = cols[col]
        df[c] = adjust_vars(df[c], 'exp', adjs.exp[col])
        df[c] = adjust_vars(df[c], 'lag', adjs.lag[col])
    return df
        
def optimize_adjustments(df=None, passes=20):
    ''' Return a table with the optimal value for each variable
        with every adjustment type. Optimal is minimization of
        the standard deviation. 
        
        df: (DataFrame) Data along with the flattened residual, and 
            the natural influences to be optimized.
            
        passes: (int) Maximum number of times to find optimal values.
                The algorithm should stop when two passes have the same
                result, but oscillations could potentially arise.
    '''
    if df is None:
        df = compile_vars()
    var_cols = 'vol solar N12 N3 N4 N34'.split()
    cols = dict(vol=['vol'], solar=['solar'], 
                enso='N12 N3 N4 N34'.split())
    vals = dict(exp=np.arange(-.1, 10.15, 0.1),
                lag=np.arange(0, 13, dtype=int))
    outcome = pd.DataFrame(index=cols.keys(), 
                           columns='exp lag result'.split())
    outcome.result = 1.0
    outcome.lag = 0
    outcome.exp = -0.1
    previous = outcome.copy()
    tests = pd.Series(index=vals['lag'], data=9.)
    original = df.flat.std()
    for p in range(passes):
        for col in cols.keys():
            ''' 
            For each natural influence `col`, smooth using the
            exponential value `exp`. Then check each lag `lag` to find
            the value with the smallest standard deviation, keeping
            other influences the same change as in `previous`. This
            is checked against the value in `outcome` and replaced
            if lower. At the end, check if `outcome` was changed from
            `previous`. If changed, copy `outcome` to `previous` and do
            another pass.
            '''
            print(col)
            vars = df[var_cols].copy()
            remain = list(cols.keys())
            remain.remove(col)
            for r in remain:  # revert remaining cols to previous best
                vars[cols[r]] = adjust_vars(df[cols[r]], 'exp',
                                            previous.exp[r])
                vars[cols[r]] = adjust_vars(vars[cols[r]], 'lag',
                                            previous.lag[r])
            for exp in vals['exp']:
                rexp = adjust_vars(df[cols[col]], 'exp', exp)
                for lag in vals['lag']:
                    vars[cols[col]] = adjust_vars(rexp, 'lag', lag)
                    c = fit(df.flat, vars[var_cols])
                    vars[var_cols] *= c[:-1]
                    residual = df.flat - vars[var_cols].sum(axis=1) - c[-1]
                    tests[lag] = residual.std()
                tests /= original
                # find best lag
                m =tests.min(skipna=True)
                mi = tests.idxmin(skipna=True)
                if m < outcome.result[col]:
                    outcome.loc[col, 'result'] = m
                    outcome.loc[col, 'lag'] = mi  # best lag
                    outcome.loc[col, 'exp'] = exp # current exp value
                tests.loc[:] = 9.
        print(f'\n==== Pass {p} ====\n')
        print(outcome)
        test = outcome.result.sum() - previous.result.sum()
        if abs(test) < .000001:
            break
        previous = outcome.copy()
    return outcome
    
def fit_vars(df=None, adjs=None, annual=False, verbose=False):
    # Important columns:
        # temp: temperature
        # lowess: local weighted trend calculated from the temperature
        # flat: temperature - lowess
        # vars: natural variation fit to the flat values
        # reduced: flat - vars
        # real: reduced + lowess
        # trend: linear trend of real
        # detrend: real - trend
    cols = dict(vol=['vol'], solar=['solar'], 
                enso='N12 N3 N4 N34'.split())
    influences = 'vol solar N12 N3 N4 N34'.split()
    if df is None:
        df = compile_vars()
    if adjs is None:
        adjs = optimize_adjustments(df)
    n = len(df)
    # adjust natural influences for best fit
    df = adjust_combined(df, adjs)
    # get trend line
    rate = 10 * 12
    xi = np.arange(n)
    m, b = np.polyfit(xi, df.temp.to_numpy(), 1)
    print(f'Original trend was {m*rate:.3}°C/decade')
    sigma = df.flat.std()
    print(f'Original standard deviation was: {sigma:.4f}°C')
    c = fit(df.flat, df[influences])
    df[influences] *= c[:-1]
    df['enso'] = df[cols['enso']].sum(axis=1)
    df['vars'] = df[influences].sum(axis=1) + c[-1]  # all variables with offset
    df['reduced'] = df.flat - df.vars
    
    # Remove periodic signal from residual
    f = np.fft.rfft(df.reduced.to_numpy())
    f = pd.DataFrame(index=np.arange(len(f)), data={'Complex': f})
    f['Value'] = np.abs(f.Complex)
    f.Value /= f.Value.max()
    f['Angle'] = np.angle(f.Complex)
    if verbose:
        ax = new_axes('FFT', 
                      'Amplitude of Frequencies in Residual', 
                      'Amplitude relative to maximum')
        ax.set_xlabel(f'Cycles per {n} Months')
        ax.plot(f.Value)
        plt.show()
    print('\nTop 2 residual frequencies:')
    fm = f.nlargest(2, 'Value')
    test = fm.Value.mean() / f.Value.mean()
    print(fm)
    print(f'Peaks over average: {test:.1f}\n')
    
    if test > 5.:  # remove periodic signals
        t = np.arange(n)
        freq = 2. * np.pi / n
        for s in fm.index:
            sname = f'Sine{s}'
            df[sname] = np.sin(freq * s * t - fm.Angle[s])
        sines = df.columns[-2:]
        c = fit(df.reduced, df[sines])
        df[sines] *= c[:-1]
        df['sine'] = df[sines].sum(axis=1) + c[-1]
        print('Sine coef:', c)
        print(f'Sine impact is ±{c.sum()/df.vars.std():.3f}σ')
        df.reduced -= df.sine
        df.vars += df.sine
    
    # calculate true trend line of reduced data
    df['real'] = df.reduced + df.lowess
    m, b = np.polyfit(xi, df.real.to_numpy(), 1)
    df['trend'] = m * xi + b
    df['detrend'] = df.real - df.trend

    nsigma = df.detrend.std()
    print(f'New standard deviation is: {nsigma:.4f}°C')
    print(f'Reduction of {(sigma-nsigma)/sigma*100:.1f}%')
    print(f'New slope is {(m*rate):.3f}°C/decade')
    r2 = tls.R2(df.flat.values, df.vars.values)
    print(f'R² value is {r2:.3f}')
    
    if annual:
        df = df.groupby(df.index.year).mean()
    return df  

def get_future(adjs):
    ''' Return a dataframe with 2016 to 2020 residual added to the end.
    '''
    df = fit_vars(adjs=adjs)
    model = df.loc['2016-01-01':'2019-12-01']
    start = '1980-01-01'
    end = '2028-01-01'
    xp = pd.date_range(start, end, freq='MS', inclusive='left')  # projection
    future = pd.DataFrame(index=xp)
    idf = df.index
    ifut = future.index[-48:]
    future.loc[idf, 'detrend'] = df.reduced.to_numpy()
    future.loc[ifut, 'detrend'] = model.reduced.to_numpy()
    m = (df.trend.iloc[-1] - df.trend.iloc[0]) / len(df)
    ixp = np.arange(len(future))
    future['trend'] = m * ixp + df.trend.iloc[0]
    future['real'] = future.trend + future.detrend
    future.spec = ''
    future.spec = df.spec
    return future
    

    
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
        start = yr - 2
        end = yr + 2
        if hasattr(data.index, 'year'):
            start = dt.datetime(start,1,1)
            end = dt.datetime(end,1,1)
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
    if hasattr(x, 'month'):
        t = f"{x:%b %y}\n{ys:.1f}σ"
    else:
        t = f"{x}\n{ys:.1f}σ"
        ax.text(x, y, t, ha='center', 
                va='bottom', size='small')
    return labels
    
# %% Plotting Functions

def plotTempTrend(df=None, adjs=None, annual=False):
    """ Plot the monthly temperature trend to 2070
    """
    def get_date(y):
        xpi = int((y - intercept) / slope)
        if xpi >= len(xp):
            return xp[-1]
        return xp[xpi]
        
    # Do analysis
    if df is None:
        df = fit_vars(adjs=adjs, annual=annual)
    # Important columns:
        # temp: temperature
        # lowess: local weighted trend calculated from the temperature
        # flat: temperature - lowess
        # vars: natural variation fit to the flat values
        # reduced: detrend - vars
        # real: reduced + lowess
        # trend: linear trend of real
        # detrend: real - trend
    
    start = df.index[0]
    if annual:
        end = 2070
        xp = np.arange(start, end+1)
    else:
        end = '2070-01-01'
        xp = pd.date_range(start, end, freq='MS', inclusive='left')  # projection
    x = df.index.values
    xi = np.arange(len(x))
    xpi = np.arange(len(xp))  # projection index
    period = {True: 'Annual', False: 'Monthly'}
    alpha = {True: .5, False: .3}
    
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
        fig = plt.figure(figname, clear=True)
        ax = fig.add_subplot(111)
        ax.set_ylim(ymin, 2.7)
        ax.plot(x, y, 'k+', alpha=alpha[annual], 
                label=f'{period[annual]} Temperature') # data
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
        text = ("Note: This is a very simplistic projection based only on past trends.\n"+
                "Natural Influences are El Niño, volcanic activity, and solar.")
        ax.text(1982, 2.2, text, size='large')
        rate = 10 * ((not annual) * 11 + 1)  # 10 or 120
        change = 'monthly'
        if annual:
            change = 'annual'
        ax.text(get_date(1.75), 1.75, f"{slope*rate:.3f}°C/decade", va='top')
        tls.titles(ax, f"Temperature Projection to 2070{title_app}",
                   f"{df.spec.name} {change} change from pre-industrial (°C)")
        tls.byline(ax)
        plt.show()
        
        return ax
         
    #=== Plot trend compared with natural influences ===
    
    y = df.temp.values
    slope, intercept = np.polyfit(xi, y, 1)
    yt = slope * xi + intercept
    ytp = slope * xpi + intercept
    ax = plot(y, yt, ytp, win_name='projection_compare',
              title_app=', Comparing Natural Influences')
    y = (df.vars + df.lowess).values
    ax.plot(df.index.values, y, label = 'Natural Influences')
    ax.legend(loc="center left")
    
    #=== Plot Trend with natural influences removed ===
    
    y = df.real.to_numpy()
    yt = df.trend.to_numpy()
    slope = (yt[-1] - yt[0]) / len(yt)
    intercept = yt[0]
    ytp = slope * xpi + intercept
    ax = plot(y, yt, ytp, win_name='projection_reduced',
              title_app=', Natural Influences Removed')
    ax.legend(loc="center left")

    plt.show()
    return

def plotTempVar(df=None, adjs=None, annual=False):
    """ Plots variability of global temperature with and without
        external variation (ENSO, volcanic sulphates, and solar) removed.
    """
    if df is None:
        df = compile_vars()
    if not hasattr(df, 'detrend'):
        df = fit_vars(df, adjs=adjs, annual=annual)
    xi = np.arange(len(df))
    slope, intercept = np.polyfit(xi, df.temp.to_numpy(), 1)
    raw_detrend = df.temp - slope * xi - intercept
    labels = max_years(raw_detrend)
    p = {True:'Annual', False:'Monthly'}[annual]
    
    # Plot temperature and Nino Index
    axs = new_fig_rows(f'Deviation-{p}',
                       "Temperature Deviation from Trend",
                       f"{df.spec.name} {p} Global Temperature (ºC)",
                       num=2)
    sigma = raw_detrend.std()
    ax = axs[0]
    plot_one(ax, raw_detrend, sigma, labels=labels)
    # plot natural influences over top
    ax.plot(df.index, df.vars.values, 'r-', lw=1)
    ax.text(df.index[-1], df.vars.iloc[-1], " Natural\n Influences",
            color='r', size='small', va='center', weight='bold')
    unit = {True: 10, False: 120}[annual]
    text = f' Trend:\n {slope*unit:.4f}°C/decade\n σ = {sigma:.3f}°C'
    ax.text(df.index[-1], -.01, text, size='small', va='top')
    ylim = ax.get_ylim()
    text = 'Natural influences are El Niño Indexes, Volcanic particles, and Solar'
    ax.text(.99, 0.01, text, ha='right', transform=ax.transAxes,
            color='r', size='small')
    # plot reduced data in next axes
    ax = axs[1]
    nsigma = df.detrend.std()
    plot_one(ax, df.detrend, nsigma, labels=labels)
    ax.set_ylim(ylim)
    ax.text(df.index[0], 2.1*nsigma, "Natural Influences Removed",
            color='k', weight='bold')
    slope = (df.trend.iloc[-1] - df.trend.iloc[0])/len(df)
    text = f' Trend:\n {slope*unit:.4f}°C/decade\n σ = {nsigma:.3f}°C'
    ax.text(df.index[-1], -.01, text,
            size='small', va='top')
   
    plt.show()
    return

def plotInfluences(df=None, adjs=None, annual=False):
    """
    Show how the fitted natural influences compare

    Parameters
    ----------
    df : pd.DataFrame, default None
        Dataframe of temperature and natural variances. Gets data from
        `compile_vars()` if not provided.
        
    adjs : pd.DataFrame, optional
        Dataframe of adjustments to make on the influences. Usually the 
        output of `optimize_adjustments()`.
        
    annual : bool, default False
        Returns annual data if true, monthly data otherwise.

    Returns
    -------
    None.

    """
    if df is None:
        df = compile_vars()
    raw = df.copy()
    df = fit_vars(df, adjs=adjs, annual=annual)
    e_cols = 'N12 N3 N4 N34'.split()
    cols = 'vol solar enso sine'.split()
    name = 'All Volcanic Solar ENSO Sine'.split()
    if 'sine' not in df.columns:
        cols.remove('sine')
        
    def scale(r, s):
        ''' Return value that scales r to s
        '''
        i = abs(s).idxmax()
        j = abs(r).idxmax()
        c = s[i] / r[j]
        corr = (s * r * c).sum()  # correct sign using correlation
        return c * np.sign(corr)

    # Scale raw data to fitted data
    for c in e_cols:
        raw[c] *= scale(raw[c], df[c])
    raw['enso'] = raw[e_cols].sum(axis=1)
    for c in cols:
        raw[c] *= scale(raw[c], df[c])
    raw['vars'] = raw[cols].sum(axis=1)
    
    # plot major influences
    axs = new_fig_rows('Influences', 
                       'Natural Influences', 
                       'Temperature Effect °C', num=len(cols)+1)
    axs[0].figure.set_size_inches(9,9)
    for i, c in enumerate(['vars']+cols):
        ax = axs[i]
        ax.plot(df[c])
        ax.plot(raw[c], alpha=0.3)
        ax.set_ylim((-.35, .35))
        ax.text(.1, .2, name[i], transform=ax.transAxes, 
                weight='bold')
    l1,  = axs[0].plot([], [], color='C1', alpha=0.3, label='Original Value')
    l2,  = axs[0].plot([], [], color='C0', label='Adjusted Value')
    ax.figure.legend(handles=[l1, l2], loc='lower left', framealpha=0.3,
                  bbox_to_anchor=(.7, .65, .5, .5))
    # Break down the ENSO index
    axs = new_fig_rows('ENSO',
                       'ENSO Components',
                       'Temperature Effect °C', num=len(e_cols)+1)
    axs[0].figure.set_size_inches(9,9)
    names = ['Combined'] + e_cols
    for i, c in enumerate(['enso']+e_cols):
        ax = axs[i]
        ax.plot(df[c])
        ax.plot(raw[c], alpha=.3)
        ax.set_ylim((-.35, .35))
        ax.text(df.index[0], -.1, names[i], weight='bold')
    l1,  = axs[0].plot([], [], color='C1', alpha=0.3, label='Original Value')
    l2,  = axs[0].plot([], [], color='C0', label='Adjusted Value')
    ax.figure.legend(handles=[l1, l2], loc='lower left', framealpha=0.3,
                  bbox_to_anchor=(.7, .69, .5, .5))
    
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
        
def plotRate(df=None, adjs=None, annual=False, stdDevs=2.0,
             verbose=False):
    """ Plot charts determining the global temperature warming rate since 1980
        and see if there is any statistically relevant acceleration.
        
        verbose: (Bool) If True, prints extra charts for blog
    """
    if df is None:
        df = compile_vars()
    if adjs is None:
        reduced = False
    if hasattr(df, 'detrend'):  # check if natural influences removed
        reduced = True
    else:
        df = fit_vars(df, adjs)
        reduced = True
    if annual:
        df = df.groupby(df.index.year).mean()
        df[yn] = df.index
    else:
        tr.convertYear(df)
    if reduced:
        df[dn] = df.real
    else:
        df[dn] = df.temp

    # get data as Numpy arrays
    x = df.Year.to_numpy()
    y = df.Data.to_numpy()
    lsq = tr.analyzeData(x, y, stdDevs)  # Analyse data
    mx, my = tr.movingAverage(x, y, 12-annual*7)  # 1-year moving average
    
    # === Plot since 1980 ===
    
    pt = {True:'Annual', False:'Monthly'}[annual]  # period text
    pi = {True:1, False:12}[annual]  # period index
    rt = {True:'(natural influences removed)', False:''}[reduced]
    
    if verbose:  # plot trend for full period
        ylabel = (f'{df.spec.name} {pt} Change from Pre-Industrial,'+
                  f'°C {rt}')
        ax = new_axes(name='Full Trend',
                      title='Temperature Trend since 1980',
                      ylabel=ylabel)
        ax.plot(mx, my, 'g-', lw=2)        # moving average
        ax.plot(lsq.xline, lsq.yline, 'b-', lw=2)  # trend
        ax.plot(x, lsq.y1, 'b-', lw=1) # lower limit
        ax.plot(x, lsq.y2, 'b-', lw=1) # upper limit
        ax.plot(x, y, 'k+', alpha=(0.3+.3*annual), lw=2)     # data
        # label chart
        error = stdDevs * lsq.sigma
        text = f'Trend: {lsq.slope*10:.3f}±{error*10:.3f} °C/decade'
        ax.text(0.5, 0.25, text, va='top', ma='left', ha='center',
                 transform=ax.transAxes)
    
    # === Plot comparison of 10 vs 20 year trend ===
    
    if verbose:
        ylabel = (f'{df.spec.name} {pt} Change from Pre-Industrial,'+
                  f'°C {rt}')
        axs = new_fig_rows('Compare Trends',
                           title='Comparing Trends with Differing Amounts of Data',
                           ylabel=ylabel,
                           num=2)
        a = df.temp.iloc[-20*pi[annual]:]  # last 20 years
        b = df.temp.iloc[-10*pi[annual]:]  # last 10 years
        for d, ax, txt in zip([a, b], axs, ['20-Year Trend', '10-Year Trend']):
            x = d.Year.to_numpy()
            y = d.Data.to_numpy()
            lsq = tr.analyzeData(x, y, stdDevs)  # Analyse data
            ax.plot(x, y, 'k+', alpha=(0.3+.3*annual), lw=1)     # data
            if not annual:
                mx, my = tr.movingAverage(x, y, 1*12)  # 1-year moving average
                ax.plot(mx, my, 'g-', lw=2)        # moving average
            ax.plot(lsq.xline, lsq.yline, 'b-', lw=2)  # trend
            ax.plot(x, lsq.y1, 'b-', lw=1) # lower limit
            ax.plot(x, lsq.y2, 'b-', lw=1) # upper limit
            # label chart
            error = stdDevs * lsq.sigma
            text = f'{txt}: {lsq.slope*10:.3f}±{error*10:.3f} °C/decade'
            ax.text(0.5, 0.15, text, va='top', ma='left', ha='center',
                     transform=ax.transAxes)

    # === Plot trends from moving breakpoint ===
    
    # set up data store for calculations
    columns = ['before', 'bhi', 'blo', 'after', 'ahi', 'alo',
               'bnu', 'bsy', 'bsxy', 
               'anu', 'asy', 'asxy']
    sides = columns[0:6:3]
    highs = columns[1:6:3]
    lows = columns[2:6:3]
    nus = columns[6::3]
    sys = columns[7::3]
    sxys = columns[8::3]
    lim = 5 * (12 - 11*annual)  # minimum number of months/years for slope
    stats = pd.DataFrame(index=df.index[lim:-lim], 
                         columns=columns, dtype=np.float64)
    stats[yn] = df.Year
    
    # calculate trend lines before and after d
    cols = [dn, yn]
    unit = 10.  # °C per decade using data in fractions of a year
    for d in stats.index:
        t1 = df.loc[df.index < d, cols]
        t2 = df.loc[df.index >= d, cols]
        for t, side, hi, lo, nu, sy, sxy in zip([t1, t2], sides, highs, lows,
                                   nus, sys, sxys):
            x = t.Year.to_numpy()
            y = t.Data.to_numpy()
            lsq = tr.analyzeData(x, y, stdDevs)  # Analyse data
            dev = stdDevs * lsq.sigma * unit
            stats.loc[d, side] = lsq.slope * unit
            stats.loc[d, hi] = lsq.slope * unit + dev
            stats.loc[d, lo] = lsq.slope * unit - dev
            stats.loc[d, nu] = lsq.nu
            stats.loc[d, sy] = lsq.sy
            stats.loc[d, sxy] = lsq.sxy
            
    # plot the before and after slopes
    pt2 = {True:'Year', False:'Month'}[annual]
    ax = new_axes(name='Slopes',
                  title=f'Comparing Trends Before and After Each {pt2}',
                  ylabel=f'{df.spec.name} Trend in °C per decade {rt}')
    ax.plot(stats.before, '-', color='C0', lw=3, label='Trend Before Date')
    ax.fill_between(stats.index, stats.blo, stats.bhi, color='C0', alpha=0.25)
    ax.plot(stats.after, '-', color='C1', lw=3, label='Trend After Date')
    ax.fill_between(stats.index, stats.alo, stats.ahi, color='C1', alpha=0.25)
    ax.set_ylim(-.2, .4)
    ax.plot([], [], 'k-', lw=10, alpha=0.15, label='95% Confidence Ranges')
    ax.legend(loc='upper center')
    
    # === plot histograms for lowest overlap ===
    
    stats['overlap'] = stats.bhi - stats.alo
    imin = stats.overlap.loc[stats.Year>2000].idxmin() 
    print(imin)
    print(stats.loc[imin])

    plt.show()
    return stats
        
def plotBreak(point, df=None, adjs=None, annual=False, stdDevs=2.):
    ''' Plot the slopes before and after a supplied break point. The point
        must be in integer or floating point years (fractional).
    '''
    if df is None:
        df = compile_vars()
    if adjs is None:
        reduced = False
    elif hasattr(df, 'detrend'):  # check if natural influences removed
        reduced = True
    else:
        df = fit_vars(df, adjs)
        reduced = True
    if annual:
        df = df.groupby(df.index.year).mean()
        df[yn] = df.index
    else:
        tr.convertYear(df)
    if reduced:
        df[dn] = df.real
    else:
        df[dn] = df.temp

    rt = {True:'(natural influences removed)', False:''}[reduced]
    rt1 = {True:'_Reduced', False:''}[reduced]
    pt = {True:'Annual', False:'Monthly'}[annual]  # period text

    ax = new_axes(name=f'Break_{pt}{rt1}_{point}',
                  title=f'Comparing {pt} Trends Before and After {point}',
                  ylabel=f'{df.spec.name} Trend in °C per decade {rt}')
    t1 = df.loc[df.Year < point, [dn, yn]]
    t2 = df.loc[df.Year >= point, [dn, yn]]
    labels = ['Before', 'After']
    colors = ['C0', 'C1']
    unit = 10  # °C/decade
    md = [0,1]  # metadata for before and after
    for t, idx, c, label in zip([t1, t2], [0,1], colors, labels):
        x = t.Year.to_numpy()
        y = t.Data.to_numpy()
        lsq = tr.analyzeData(x, y, stdDevs)  # Analyse data
        md[idx] = lsq
        x = [t.Year.iloc[0], t.Year.iloc[-1]]
        y = [lsq.slope * unit, lsq.slope * unit]
        ax.plot(x, y, '-', color=c, lw=2, label=label)
        err = lsq.sigma * stdDevs * unit
        ax.fill_between(x, y-err, y+err, color=c, alpha=0.15)
    tr.analyzeRate(df, dn, window=20, stdDevs=stdDevs)
    x = df.Year.to_numpy()
    y = df.Rate.to_numpy() * unit
    ax.plot(x, y, 'k-', lw=1, label='Rate with 20-year window')
    ax.fill_between(x, df.R1*unit, df.R2*unit, color='k', alpha=.05)
    ax.set_ylim(0, .6)
    ax.plot([], [], 'k-', lw=10, alpha=0.15, label='95% Confidence Ranges')
    ax.legend(loc='upper left')
    
    # === Plot temperature with these trends ===
    
    change = {True: 'annual', False: 'monthly'}[annual]
    ax = new_axes(name=f'Slopes_{pt}{rt1}_{point}',
                  title=f'Comparing {pt} Trends Before and After {point:.0f}',
                  ylabel=f'{df.spec.name} {change} change from pre-industrial (°C) {rt}')
    md[0].xline[1] = df.Year.iloc[-1]
    for lsq, c in zip(md, colors):
        ax.plot(lsq.x, lsq.y1, '-', color=c, lw=0.5)
        ax.plot(lsq.x, lsq.y2, '-', color=c, lw=0.5)
        yline = lsq.slope * lsq.xline + lsq.intercept
        ax.plot(lsq.xline, yline, '-', color=c, lw=2)
    ax.plot(df.Year.values, df.Data.values, 'k+', alpha=0.4)
    lsq = tr.analyzeData(df.Year.to_numpy(),
                         df.Data.to_numpy(),
                         stdDevs)
    c = chow(lsq.res, md[0].res, md[1].res, 2)
    hyp = {True:'Pass above', False:'Fail below'}[(c.p > 0.05)]
    text = ('Chow test for no break: \n' +
            f'Test: {c.Chow:0.1f} (DF₁: {c.DoF1:.0f}, DF₂: {c.DoF2:.0f}\n' +
            f'p-value: {c.p:0.3f} ({hyp} 0.05 threshold)')
    ax.text(.2, .7, text, transform=ax.transAxes,
            ha='left')

    plt.show()

