#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
defs useful for calculating trends and their confidence levels

Ported from Javascript
(c) Skeptical Science 2012
Attribution-ShareAlike 3.0 Unported (CC BY-SA 3.0)
https://creativecommons.org/licenses/by-sa/3.0/

This module accepts Pandas DataFrames with datetime indexes for data

@author: dan
"""

import pandas as pd
import numpy as np

class Holder:
    """ Simple class to hold data as attributes.
    """

def convertYear(data):
    """ Add a 'Year' column, if it does not already exist, containing
        the fractional year. `data` must have a pd.datetime index.
        This method accounts for leap years.
    """
    if 'Year' in data.columns: return
    if not hasattr(data.index, 'year'):  # integer year index
        data['Year'] = data.index
        return
    dy = data.index.year
    doy = data.index.day_of_year - 1
    ey = data.index.is_leap_year * 1 + 365
    data['Year'] = dy + doy / ey

def selectData(data, cn=None, start=None, end=None):
    """ Extract xy data from the temperature data object by date range
            data:  (Pandas DataFrame) with an index of datetime
            cn:    (string) Column name to use. Default is first one
            start: (float) start date as fractional year
            end:   (float) end date as fractional year

            Returns data as x and y DataSeries
    """
    if not start: start = -np.inf
    if not end: end = np.inf
    if not cn: cn = data.columns[0]
    convertYear(data)
    d = data.loc[data['Year'].between(start, end)]
    return d['Year'], d[cn]

def linearFit(xdata, ydata):
    """ Determines the linear fit to supplied Numpy arrays. Use
        `df['Column Name'].values` to extract array from DataFrame.
        Returns dictionary of results.
        xdata, ydata: (numpy array) Input x and y data
    """
    # accumulate sums
    n = len(xdata)
    sx = xdata.mean()
    sy = ydata.mean()
    sxx = (xdata*xdata).mean()
    sxy = (xdata*ydata).mean()
    # trend
    if sxx > sx*sx:
        b = (sxy - sx*sy)/(sxx - sx*sx)
    else: b=0
    a = sy - b*sx
    # uncertainty
    sd2 = ((ydata - (a+b*xdata))**2).sum()/(n - 2)  # residual variance
    sb2 = sd2/(n*(sxx-sx*sx))  # slope variance
    sa2 = sxx*sb2  # residual variance
    # package results
    lsq = Holder()  # empty object
    lsq.n = n
    lsq.sx = sx
    lsq.sy = sy
    lsq.sxx = sxx
    lsq.sxy = sxy
    lsq.slope = b
    lsq.intercept = a
    lsq.slopeVar = sb2
    lsq.interceptVar = sa2
    lsq.residualVar = sd2
    lsq.x = np.array([xdata.min(), xdata.max()])
    lsq.y = a + b * lsq.x
    return lsq

def dataPerDegreeOfFreedom(xdata, ydata, lsq=None):
    """ Returns the number of data points per degree of freedom
        based on the covariance of the data. See:
          Foster and Rahmstorf, 2011
          "Global temperature evolution 1979–2010"
          https://iopscience.iop.org/article/10.1088/1748-9326/6/4/044022#erl408263app1

        xdata, ydata: (numpy array) Input x and y data
    """
    if not lsq:
        lsq = linearFit(xdata, ydata)
    yres = ydata - lsq.slope * xdata
    cov = autocovariance(yres, 0)
    rho1 = autocovariance(yres, 1) / cov
    rho2 = autocovariance(yres, 2) / cov
    nu = 1.0 + (2.0*rho1)/(1.0-rho2/rho1)
    if nu < 0.:
        nu = (1. + rho1)/(1. - rho1)
    return nu

def autocovariance(data, j):
    """ Return covariance of the data at lag j
        data: (numpy array) data to be analyzed
        j:    (int) lag to calculate the covariance at
    """
#  var n = data.length, sx = 0.0, cx = 0.0;
#  for ( var i = 0; i < n; i++ )
#    sx += data[i];
#  sx /= n;
#  for ( var i = 0; i < n-j; i++ )
#    cx += (data[i]-sx)*(data[i+j]-sx);
#  return cx/n;
    n = len(data)
    sx = data.mean()
    cx = ((data[:n-j] - sx) * (data[j:] - sx)).mean()
    return cx

def movingAverage(xdata, ydata, period):
    """ Return the moving average of the supplied data. Assumes
        that the data is at regular intervals.
        xdata, ydata: (numpy array) data to be analyzed
        period:       (int) number of values per average
        Returns x, y: (lists of floats)
    """
    x = []
    y = []
    for i in range(len(xdata)-period+1):
        x.append(xdata[i:i+period].mean())
        y.append(ydata[i:i+period].mean())
    return  x, y

def confidenceInterval(xdata, ydata, sigma, lsq=None):
    """ Calculate the slope limits for the supplied confidence limit.
        xdata, ydata: (numpy array) data to be analyzed
        sigma:        (float) Value of deviation to use for interval
        lsq:          (object) Optional least squares data. Will calculate
                      if not supplied.
        Returns lsq with y1 and y2 added
    """
    if not lsq:
        lsq = linearFit(xdata, ydata)
    y = lsq.intercept + lsq.slope * xdata
    xvar = (xdata - lsq.sx)**2
    dy = sigma * (lsq.residualVar * (1 + xvar/(lsq.sxx-lsq.sx**2))/lsq.n)**0.5
    lsq.y1 = y - dy
    lsq.y2 = y + dy
    return lsq

def analyzeData(xdata, ydata, stdDevs=2.):
    """ Get the trend of the supplied data, along with the confidence
        interval at the supplied number of standard deviations.
        xdata, ydata: (numpy array) data to be analyzed
        stdDevs:      (float) Number of standard deviations for confidence
                      interval. Default is 2.0 (95%).
        Returns lsq with analysis values and confidence limits y1, y2
    """
    if hasattr(xdata, 'values'): x = xdata.values
    else: x = xdata
    if hasattr(ydata, 'values'): y = ydata.values
    else: y = ydata
    lsq = linearFit(x, y)
    nu = dataPerDegreeOfFreedom(x, y, lsq)
    nu = max(nu, 0)
    lsq.sigma = (nu * lsq.slopeVar)**0.5
    lsq.nu = nu
    lsq = confidenceInterval(x, y, stdDevs * nu**0.5, lsq)
    return lsq

def analyzeRate(df, cn, window, stdDevs=2.):
    """ Analyze the slope at each point in the supplied data, and
        include the adjusted variance.
        df:           (Pandas DataFrame) data to analyzed
        cn:           (string) column name to analyze
        window:       (float) size of the window in years, centred on
                      each point, to use for calculating the slope.
        stdDevs:      (float) Number of standard deviations for confidence
                      interval. Default is 2.0 (95%).
    """
    yn = 'Year'
    if yn not in df.columns:
        convertYear(df)
    w2 = window/2
    for i, yr in zip(df.index, df[yn]):
        d = df.loc[df[yn].between(yr-w2, yr+w2)]
        x = d[yn].values
        y = d[cn].values
        lsq = linearFit(x, y)
        nu = dataPerDegreeOfFreedom(x, y, lsq)
        sigma = (nu * lsq.slopeVar)**0.5 * stdDevs
        df.loc[i, 'Rate'] = lsq.slope
        df.loc[i, 'Rate-'] = lsq.slope - sigma
        df.loc[i, 'Rate+'] = lsq.slope + sigma

def example():
    import matplotlib.pyplot as plt

    # Create trend line with autocorrelated noise
    w = np.ones(21)/21. # filtering window
    d = pd.DataFrame(index=pd.date_range(start='1980-1-1',
                                         end='2020-1-1',
                                         freq='MS'))
    d['n'] = np.random.normal(0, 2., len(d))
    d.n = np.convolve(d.n, w, 'same') # filter with a moving average
    d.n -= d.n.mean()
    d['ya'] = 0.01
    d.ya = d.ya.cumsum()
    d['y'] = d.ya + d.n

    # Do analysis
    stdDevs = 2.
    x, y = selectData(d, 'y', start=1990.5)
    lsq = analyzeData(x, y, stdDevs)
    mx, my = movingAverage(x, y, 12)

    #save data for inspection
    lsq.xdata = x
    lsq.ydata = y

    fig, ax = plt.subplots(num='example', clear=True)

    ax.plot(x, y, 'k+', alpha=0.3)     # data
    ax.plot(mx, my, 'g-', lw=2)        # moving average
    ax.plot(lsq.x, lsq.y, 'b-', lw=3)  # trend
    ax.plot(x, lsq.y1, 'b-', lw=1) # lower limit
    ax.plot(x, lsq.y2, 'b-', lw=1) # upper limit

    # add labels with info
    error = stdDevs * lsq.sigma
    text = (f'Actual: 0.12/year\nTrend: {lsq.slope:.2f}±{error:.3f}\n'+
            f'nu: {lsq.nu:.3f}')
    ax.text(0.5, 0.25, text, va='top', ma='left', ha='center',
             transform=ax.transAxes)
    plt.show()
    return lsq

if __name__ == '__main__':
    lsq = example()

