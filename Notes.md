# Experiment Notes

## 2024-03-04

* Significant work done:

    * New annual data approach should work with all plot functions
    
    * Added a Chow Test to assess breaks in data
    
    * still looking at how the `nu` value works. Clipped it to 1.0 
      whenever it goes below that amount. This prevents the variance
      from decreasing below the calculated value.
      
    * Added a simple exponential function to model a more gradual
      warming.

### Next:

* Add the ability to use the simple exponential function for all 
  variables (find optimum value in advance for each).
  
* Add the ability to just use lags (as per Foster and Rahmstorf 2011).
  Again find the optimum value in advance for each).

## 2024-02-28

* Removed periodic signal

* Changed fitting to only work on monthly data, then convert to annual if
  required.
  
* Changed fitting to use lowess smoothing for removing trend, instead of a
  straight line. Lowess smoothing uses locally weighted slopes for each
  point of data.

## 2024-02-28

* Added smoothing to ENSO values.

* Consider making smoothing function an optimized exponential like Tomino

* Check residual for any periodic signal with FFT as in Foster and 
  Rahmstorf, 2011

## 2024-02-27

* Fixed some persistent plotting problems with monthly data

* 2002 produces a really small `nu` value before and after. This makes the 
  estimated error very small. Must investigate further.

## 2024-02-22

* Made the charts include annual and reduced data. Also had both slopes (before
  and after the breakpoint) include the breakpoint. Before, the before slope did 
  not include it. The downside of this is that outliers will exacerbate the 
  differences in the two slopes, bringing them in opposite directions.

## 2024-02-21

* New priority: see if the temperature is accellerating.

## 2024-02-20

* started a plotting function to explain how the ocean warming curve is applied.

## 2024-02-15

* The pre-industrial era (PIE) offset changed  
    * from 0.3117  
    * to 0.366609  
  This is the the offset from 1850-1899 to 1961-1990 in Hadcrut5. Other termperature
  sources are normalized with this value. I.e. all datasets are normalized by
  subracting the mean of 1961-1990, then adding 0.366609 so than now the mean
  of 1961-1990 = 0.366609 for all sets.
  
* The plots were finalized for posting.

## 2024-02-14

* Made annual version of chart. Has much better R^2 value.

* Solar has a long-term trend which should be in the projection. But since the
trend is currently -ve, it actually moves the crossing points further out. But
it might not be that, so will leave the more pessimistic values, which are still
more optimistic that the simple projection.

### Next:

* Make plots postable (legend, etc)

## 2024-02-13

* Combining the axes onto one figure created a very squished image, even when
stretched vertically to fit the entire sceen. The headings and text would all
need to be adjusted. In the end, it is easier to composite the two charts onto
one image manually.

* Solar trend was removed from the fit data. Any trend would show up in the
linear fit, so this prevents doubling and makes a better fit to detrended 
data.

* Data with solar trend removed:  
> Original standard deviation was: 0.1411°C  
> New standard deviation is: 0.1133°C  
> Reduction of 19.7%  
> New slope is 0.191°C/decade  
> R² value is 0.351  

### Next:

* Make a year-end annual chart

## 2024-02-12

* Tried squaring ENSO (actually enso * enso.abs()) to get better matching on
peaks, but the overall variance reduction went from 19.7% to only 18.0%

* The Pacific Decadal Oscillation has no effect on the overall variance
reduction.

* MEI had a much worse fit than all the ENSO indices together

### Next:

* Move the axes onto one figure for easier posting.
