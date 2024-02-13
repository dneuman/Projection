# Experiment Notes

## 2024-02-13

* Combining the axes onto one figure created a very squished image, even when
stretched vertically to fit the entire sceen. The headings and text would all
need to be adjusted. In the end, it is easier to composite the two charts onto
one image manually.

* Data:  
> Original standard deviation was: 0.1411°C  
> New standard deviation is: 0.1133°C  
> Reduction of 19.7%  
> New slope is 0.191°C/decade  
> R² value is 0.351  

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
