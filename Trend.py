#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
defs useful for calculating trends and their confidence levels

Ported from Javascript
https://skepticalscience.com/trend.php
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
    
def padded(s, size, ptype='linear'):
    """Takes a series and returns a version padded at both ends.

    Parameters
    ----------
    s : pd.Series
        Series to be padded
    size : int
        Size of window being used. The returned series will be
        (size - 2) bigger than the supplied series.
    type : str ['linear' | 'mirror'] default 'linear'
        Type of padding to use. 'Linear' fits a line to the end data of length
        ``size`` and uses that to fill the start and end padding. 'Mirror'
        copies and reflects the data instead.

    Notes
    -----
    'mirror' uses actual data for padding, but results in zero-slope
    (horizontal) end points. 'linear' will usually give better results.
    """

    if type(s) != pd.Series:  # Check for dataframe
        s = s.iloc[:, 0]  # use first column as a series
    n = len(s)
    hw = size//2  # half-window size
    tx = np.array(s.index)
    ty = np.array(s.values)
    x = np.zeros(n + 2 * hw, dtype=s.index.dtype)
    y = np.zeros(n + 2 * hw)
    x[hw:hw+n] = tx  # add actual data
    y[hw:hw+n] = ty

    # x-value intervals are mirrored in both cases
    for i in range(hw):  # pad beginning
        x[i] = tx[0] - (tx[hw-i] - tx[0])
    for i in range(hw):  # pad end
        x[i+hw+n] = tx[n-1] + (tx[n-1] - tx[n-2-i])

    if ptype.lower() == 'mirror':
        # pad data as a reflection of original data. eg use index values:
        # 2, 1, 0, 1, 2, 3, 4, 5 and
        # n-3, n-2, n-1, n-2, n-3, n-4
        for i in range(hw):  # pad beginning
            y[i] = ty[hw-i]
        for i in range(hw):  # pad end
            y[i+hw+n] = ty[n-2-i]
    else:
        # use 'linear' for any other input
        # Note that x values may be dates, so normalize them so fit line
        # coefficients are not too large
        ix = x.copy()
        if type(x[0]) == np.datetime64:
            # make days, then floats
            tx = (tx - x[0]).astype('m8[D]').astype(np.float64)
            ix = (ix - x[0]).astype('m8[D]').astype(np.float64)
        # fit a line to start
        c = np.polyfit(tx[:hw], ty[:hw], 1)
        p = np.poly1d(c)
        y[:hw] = p(ix[:hw])
        # fit a line to end
        c = np.polyfit(tx[-hw:], ty[-hw:], 1)
        p = np.poly1d(c)
        y[-hw:] = p(ix[-hw:])

    return pd.Series(y, index=x)   
 
def lowess(data, f=2./3., pts=None, itns=3, order=1,
           pad='linear', **kwargs):
    """ Locally-Weighted Slope Smoothing. Fits a nonparametric regression 
        curve to a scatterplot.

    Parameters
    ----------
    data : pandas.Series
        Data points in the scatterplot. The
        function returns the estimated (smooth) values of y.
    f : float default 2/3
        The fraction of the data set to use for smoothing. A
        larger value for f will result in a smoother curve.
    pts : int default None
        The explicit number of data points to be used for
        smoothing instead of f.
    itn : int default 3
        The number of robustifying iterations. The function will run
        faster with a smaller number of iterations.
    order : int default 1
        The order of the polynomial used for fitting. Defaults to 1
        (straight line). Values < 1 are made 1. Larger values should be
        chosen based on shape of data (# of peaks and valleys + 1)
    pad : str ['linear' | 'mirror' | None] default 'linear'
        Type of padding to use. If no padding desired, use ``None``.

    Returns
    -------
    pandas.Series containing the smoothed data.

    Notes
    -----
    Surprisingly works with pd.DateTime index values.
    """
    # Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
    #            original
    #          Dan Neuman <https://github.com/dneuman>
    #            converted to Pandas series, extended to polynomials,
    #            and added padding option.
    # License: BSD (3-clause)

    n = len(data)
    if pts is None:
        f = np.min([f, 1.0])
        r = int(np.ceil(f * n))
    else:  # allow use of number of points to determine smoothing
        r = int(np.min([pts, n]))
    r = min([r, n-1])
    order = max([1, order])
    if pad:
        s = padded(data, r*2, ptype=pad)
        x = np.array(s.index)
        y = np.array(s.values)
        n = len(y)
    else:
        x = np.array(data.index)
        y = np.array(data.values)
    # condition x-values to be between 0 and 1 to reduce errors in linalg
    x = x - x.min()
    x = x / x.max()
    # Create matrix of 1, x, x**2, x**3, etc, by row
    xm = np.array([x**j for j in range(order+1)])
    # Create weight matrix, one column per data point
    h = [np.sort(np.abs(x - x[i]))[r] for i in range(n)]
    w = np.clip(np.abs((x[:, None] - x[None, :]) / h), 0.0, 1.0)
    w = (1 - w ** 3) ** 3
    # Set up output
    yEst = np.zeros(n)
    delta = np.ones(n)  # Additional weights for iterations
    for iteration in range(itns):
        for i in range(n):
            weights = delta * w[:, i]
            xw = np.array([weights * x**j for j in range(order+1)])
            b = xw.dot(y)
            a = xw.dot(xm.T)
            beta = np.linalg.solve(a, b)
            yEst[i] = sum([beta[j] * x[i]**j for j in range(order+1)])
        # Set up weights to reduce effect of outlier points on next iteration
        residuals = y - yEst
        s = np.median(np.abs(residuals))
        delta = np.clip(residuals / (6.0 * s), -1, 1)
        delta = (1 - delta ** 2) ** 2
    if pad:
        n = len(data)
        return pd.Series(yEst[r:n+r], index=data.index,
                         name='Locally Weighted Smoothing')
    else:
        return pd.Series(yEst, index=data.index,
                         name='Locally Weighted Smoothing')


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

def linearFit(x, y):
    """ Determines the linear fit to supplied Numpy arrays. Use
        `df['Column Name'].values` to extract array from DataFrame.
        Returns dictionary of results.
        xdata, ydata: (numpy array) Input x and y data
    """
    # accumulate sums
    n = len(x)
    sx = x.mean()
    sy = y.mean()
    sxx = (x*x).mean()
    sxy = (x*y).mean()
    # trend
    if sxx > sx*sx:
        b = (sxy - sx*sy)/(sxx - sx*sx)
    else: b=0
    a = sy - b*sx
    # uncertainty
    sd2 = ((y - (a+b*x))**2).sum()/(n - 2)  # residual variance
    sb2 = sd2/(n*(sxx-sx*sx))  # slope variance
    sa2 = sxx*sb2  # residual variance
    # package results
    lsq = Holder()  # empty object
    lsq.x = x
    lsq.y = y
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
    lsq.xline = np.array([x.min(), x.max()])
    lsq.yline = a + b * lsq.xline
    lsq.res = lsq.y - (a + b * x)
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
    rx = data - data.mean()
    cx = (rx[:n-j] * rx[j:]).mean()
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

def confidenceInterval(xdata, ydata, stdDevs, lsq=None):
    """ Calculate the slope limits for the supplied confidence limit.
        xdata, ydata: (numpy array) data to be analyzed
        stdDevs:      (float) number of standard deviations to use for interval
        lsq:          (object) Optional least squares data. Will calculate
                      if not supplied.
        Returns lsq with y1 and y2 added
    """
    if not lsq:
        lsq = linearFit(xdata, ydata)
    y = lsq.intercept + lsq.slope * xdata
    xvar = (xdata - lsq.sx)**2
    dy = stdDevs * (lsq.residualVar * (1 + xvar/(lsq.sxx-lsq.sx**2))/lsq.n)**0.5
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
    nu_adj = max(1., nu)
    lsq.sigma = (nu_adj * lsq.slopeVar)**0.5
    lsq.nu = nu
    lsq = confidenceInterval(x, y, stdDevs * nu_adj, lsq)
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
        nu_adj = max(1., nu)
        sigma = (nu_adj * lsq.slopeVar)**0.5 * stdDevs
        df.loc[i, 'Rate'] = lsq.slope
        df.loc[i, 'R1'] = lsq.slope - sigma
        df.loc[i, 'R2'] = lsq.slope + sigma

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

