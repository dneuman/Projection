# Projection
 Estimate of a "Business as Usual" Climate Projection
 
## Introduction  
This model uses recent trends in emissions to estimate future impact on climate. It first estimates climate sensitivity from past atmospheric greenhouse gas (GHG) concentrations and other climate forcings, using a simplified ocean warming model. It then estimates CO~2 sinks based on past emissions using the same ocean warming model. Finally, it applies a series of assumptions on future emissions based on current market trends.

## Assumptions    
- **Renewable Energy** will be estimated by a logistic curve fitted to existing growth to reach 100% of market. This will have significant uncertainty.  
- **Fossil Energy** will be the remaining energy requirement. Coal, Oil and gas will each trend within the market share as in the recent past.  
- **Energy Demand** will decrease with the increase in renewable energy, with the assumption that the electrification of demands will cause more efficient use of energy, either at the electricity generation plant or for internal combustion engines.  
- **Agriculture Emissions** will trend as is.  
- **Concrete Emissions** will trend as is.
- **Industrial Emissions** will trend as is.
- **Air Travel and Ocean Shipping** will trend as is, except a portion of ocean shipping will decrease with fossil fuel production.  
- **Methane Emissions** will trend with fossil energy production, except for the portion produced by agriculture.  
- **Pollution** will trend with fossil energy production.  
- **Land Use** will be unchanged.  
- **CFCs and other GHGs** will continue trend.

## Data  
Data will be downloaded from the source where possible. The `specs.yaml` file contains details of where each file is from, and how that file was imported.