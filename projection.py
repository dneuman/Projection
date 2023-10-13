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

dst = ds.dst
plt.style.use('clean')

yn, mn, dn, en, sn = ['Year', 'Month', 'Data', 'Error', 'Smooth']


def plotTempTrend():
    """
    """
    # Do analysis
    
    source = 'hadcrut'
    ds.update_modern(source)
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
    slope, intercept = tls.ts_est(df.Data)  # get median trend
    # median trend has same number of points above the line as below it
    ys = slope * xi + intercept
    ysp = slope * xpi + intercept
    
    dy = y - ys  # detrend the data
    sigma = dy.std()
    
    def get_date(y):
        xpi = int((y - intercept) / slope)
        return xp[xpi]
        
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
    
    ax.text(x[-1], y[-1], f'  {df.index[-1]:%b %y}', 
            ha='left', va='center', size='small')  # last month
    lastx = df.loc[df.index.year<2020, 'Data'].idxmax()
    lasty = df.loc[lastx, 'Data']
    ax.text(lastx, lasty, f'{lastx:%b %y}  ', 
            ha='right', va='center', size='small')  # last month

        
    for k in dates.keys():
        ax.hlines(k, xmin, dates[k], color='k', lw=0.5, ls=':')
        ax.vlines(dates[k], ymin, k, color='k', lw=0.5, ls=':')
        ax.text(dates[k], k, dates[k].year, ha='left', va='bottom', weight='bold')

    ax.text(xp[-1], ysp[-1]+sigma*2, '95% Range')
    ax.text(xp[-1], ysp[-1]+sigma, '68% Range')
    ax.text(xp[24], 2.2, "Note: This is a very simplistic projection based "+ \
            "only on past trends", size='large')

    tls.byline(ax)
    tls.titles(ax, "Temperature Projection to 2070",
               f"{spec.name} monthly change from pre-industrial (°C)")
    plt.show()
    return

plotTempTrend()