#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic tools for plotting and data processing

@author: dan613
"""
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import numpy as np
from scipy.interpolate import UnivariateSpline as Spline

yn, mn, dn, en, nn, sn = ['Year', 'Month', 'Data',
                          'Error', 'Normalized', 'Smooth']
emn, epn = ['Err-', 'Err+']  # difference from mean/median
e5, e95 = ['5', '95']
sln, sdn = ['Slope', 'Deviation']
intn, fn = 'Integral', 'Forcing'

#%% Graphing tools
def byline(ax: plt.axes, x=0.01, y=0.01,
           ha='left', va='bottom', **kwargs) -> plt.text:
    """
    Add a byline to the supplied axes. Default location is lower left.

    Parameters
    ----------
    ax: axes to byline
    x: x value in axes coordinates (0.-1.)
    y: y value in axes coordinates (0.-1.)
    ha: horizontal alignment
    va: vertical alignment
    kwargs: passed to the ax.text() method

    Returns
    -------
    ax.text object
    """
    d = dt.datetime.now()
    date = d.strftime('%d %b %Y')
    text = 'Chart: mstdn.ca/@dan613 ©CC-BY 4.0' + '  ' + date
    if ha=='right' and x==0.01:
        x = 0.99
    if va=='top' and y==0.01:
        y = 0.99
    txt = ax.text(x, y, text, fontsize='small', ha=ha, va=va,
                   transform=ax.transAxes, **kwargs)
    return txt

def wrap(ax: plt.Axes, text: str, x: float, y: float, w: int, **kwargs):
    """ Wrap text in the supplied axes to width w in pixels

        The default axes is ax.transAxes.
    """
    trn = 'transform'
    trnames = dict(data=ax.transData, axes=ax.transAxes)
    if not isinstance(text, str):
        raise Exception('tools.wrap interface has changed. Check code.')
    if trn not in kwargs:
        if abs(x)>1.5 or abs(y)>1.5:
            tr = ax.transData
        else:
            tr = ax.transAxes
        kwargs[trn] = tr
    if isinstance(kwargs[trn], str):
        if kwargs[trn] not in trnames:
            raise Exception('Unknown transform in tools.wrap. '+ \
                            'Use "data" or "axes" only.')
        kwargs[trn] = trnames[kwargs[trn]]
    t = ax.text(x, y, text, wrap=True, **kwargs)
    t._get_wrap_line_width = lambda: w
    return t

def titles(ax: plt.Axes, t1: str, t2: str):
    """ Put a title and a subtitle at top of axes. Works with
        one axes per figure, otherwise may be crowded.
    """
    tr = ax.transAxes
    rc = plt.rcParams
    f = 'figure.'
    a = 'axes.'
    w = 'titleweight'
    s = 'titlesize'
    txt1 = ax.text(0.0, 1.05, t1, ha='left', va='bottom', transform=tr,
                   size=rc[f+s], weight=rc[f+w])
    txt2 = ax.text(0.0, 1.04, t2, ha='left', va='top', transform=tr,
                   size=rc[a+s], weight=rc[a+w])
    return txt1, txt2

def labels(ax: plt.Axes, title: str, xlabel: str, ylabel: str):
    """ Fill in the title, x and y labels for the supplied axes. The x 
        label will be positioned under the title. Change the style
        figure.titleweight and .titlesize to adjust title.
    """
    tr = ax.transAxes
    rc = plt.rcParams
    f = 'figure.'
    w = 'titleweight'
    s = 'titlesize'
    rt = ax.set_title(title, loc='left', ha='left', va='bottom',
                   size=rc[f+s], weight=rc[f+w])
    ry = ax.text(0.0, 1.002, ylabel, ha='left', va='bottom', transform=tr)
    rx = ax.set_xlabel(xlabel)
    return rt, rx, ry


def Annotate_Line(ax, text, xy, offsetpts=20, height=.05, above=True):
    """ Annotate using a vertical line to demark position with a horizontal
        line pointing to text.
    """
    b = ax.get_ybound()
    length = b[1] - b[0]
    lx = [xy[0], xy[0]]
    if not above: height *= -1.
    ly = [xy[1], xy[1]+length*height]
    ha = 'left'
    if offsetpts < 0: ha = 'right'
    ax.plot(lx, ly, color='k', lw=1.5)
    arrow = dict(color='k', width=1, headwidth=1., headlength=1)
    ax.annotate(text, (xy[0], xy[1]+length*height*0.5), xytext=(offsetpts, 0),
                textcoords='offset points', va='center', ha=ha,
                arrowprops=arrow, wrap=True)

def addColor(ax, c):
    """ Add color to the y axis
    """
    ax.tick_params(axis='y', labelcolor=c)
    ax.yaxis.label.set_color(c)

#%% Data tools

def gaussian(n=50, lim=3):
    """ Gaussian window, with endpoints near 0.
    """
    x = np.linspace(-lim, lim, n)
    s = -0.5 * x**2
    y = np.exp(s)
    y -= y[0] - .01
    y /= y.max()
    return y

def loess(s: pd.Series, winlen: float, windowed=True,
          error: pd.Series = None) -> pd.DataFrame:
    """
    Return DataFrame with smoothed data and fit range.

    s: Series with monotonically increasing date for index
    winlen: float Length of window in years or fractional years
    error: Optional Series with average 1 sigma error for each row of s
"""
    if not error and not windowed:
        print('\nError values required if not using gaussian window')
        windowed = True
    yweights = gaussian(2 * winlen + 1)
    # make a spline of the weights to calculate intermediate values
    xweights = np.arange(-winlen, winlen+1)
    spl = Spline(xweights, yweights, s=0)
    r = pd.DataFrame(index=s.index, columns=[sn, sln, sdn])
    sx = s.copy()
    ix = sx.index
    if hasattr(s.index, 'dayofyear'):
        sx.index = ix.year + ix.dayofyear/(365+ix.is_leap_year * 1)
    for i, yr in enumerate(sx.index):
        span = sx.loc[yr - winlen/2:yr + winlen/2]
        x = span.index.values.astype('float')
        y = span.values
        # assume uneven time used. Must interpolate weights for each x
        if windowed:
            weights = spl(x-yr)
        else:
            err = error.loc[span.index]
            weights = 1./err.values
        [[slope, intercept], cov] = np.polyfit(x, y, 1, w=weights, cov=True)
        r[sn].iloc[i] = (yr * slope + intercept)
        r[sln].iloc[i] = slope
        r[sdn].iloc[i] = cov[0, 0]**.5
    return r

def smooth(s: pd.Series, winlen: int = None, window: str = None,
           pad: str = None) -> pd.DataFrame:
    """
    Smooth data with a moving average, using a gaussian window.

    Parameters
    ----------
    s : pd.Series
        DESCRIPTION.
    winlen : int, optional
        Length of window. Default is None.
    window : String, optional
        [None|'gauss'|'hamming']. The default is None.
    pad : String, optional
        ['zero'|'value']. The default is zero.

    Returns
    -------
    Dataframe with smoothed data.

    """
    n = len(s)
    if not winlen:
        winlen = n//5
    if window == 'gauss':
        win = gaussian(winlen)
    elif window == 'hamming':
        win = np.hamming(winlen)
        win = win - win[0] + 0.01
    else: win = np.ones(winlen) / winlen  # square window
    
    win /= win.sum()
    winlen += winlen//2 * 2 + 1  # make winlen odd
    w2 = winlen//2
    
    p = np.zeros(n+w2*2)
    p[w2:-w2] = s.values
    if pad=='value':
        p[:w2] = s.iloc[0]
        p[-w2:] = s.iloc[-1]
    
    r = np.convolve(p, win, 'same')
    dr = pd.Series(index=s.index,
                   data=r[w2:-w2], dtype=float)
    return dr

def ts_est(ds):
    """ The Thiel-Sen estimator takes the median of slopes
        between all points, then takes the median of the
        y-intercept of all points using the calculated slope.
        It is resistant to outliers.

        ds: Pandas Series with monotonically increasing index (date).
    """
    x = np.arange(len(ds))
    y = ds.values
    n = len(x)
    i = np.ones((n, n))
    iu = np.triu_indices(n, 1)  # get upper right indices
    # calculate slope between each point (one direction only)
    mx = x * i
    my = y * i
    tx = (mx - mx.T)[iu]
    ty = (my - my.T)[iu]
    slope = np.median(ty / tx)
    intercept = np.median(y - slope * x)
    return slope, intercept

def median_sm(ds, winlen, passes=5):
    """ Return smoothed line similar to loess, but using the
        Thiel-Sen estimator.
    """
    rs = ds.copy()
    rs.sort_index(inplace=True)
    for p in range(passes):
        cs = rs.copy()
        for x in cs.index:
            d = cs.loc[x-winlen:x+winlen]
            slope, intercept = ts_est(d)
            rs[x] = slope*x + intercept
    return rs

def R2(data, estimate):
    """ Return R² value for arrays data and its estimate
    """
    e = data.mean()
    return 1 - np.square(estimate - data).sum() / np.square(data - e).sum()

def get_offset(first, last):
    """ Return offset required to align the supplied dataframes
        on the overlapping periods.
    """
    ymax = last[yn].max()
    ymin = first[yn].min()
    lastmean = last[dn].loc[last[yn].between(ymin, ymax)].mean()
    firstmean = first[dn].loc[first[yn].between(ymin, ymax)].mean()
    offset = firstmean - lastmean
    return offset