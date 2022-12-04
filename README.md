# Berkely-Temp-Analysis

The purpose of this project is to explore the climate trends and build a predictive model.

## Goals of the Project

1. Explore a climate change dataset with Xarray framework.
2. Data manipulation to generate insights on climate change around the globe and USA.
3. Visualize the dataset with various plot types.
4. Prepare the data for time-series prediction
5. Build ARIMA and Prophet Models


## Materials and methods:

This project involved two datasets. The data that we are going to use for this is gathered from Berkely Earth: http://berkeleyearth.org/data/ and CRU https://crudata.uea.ac.uk/cru/data/temperature/. This dataset is publicly available for research. As part of the work, the task of analyzing climate change and its trend is required. For the final part we build a predictive model to predict the future trend.


## File Structure:

Analysis.ipynb: Analyzed the netCdf dataset using xarray and visualtion using cartopy and hvplot.
Untitled.ipynb: File for ARIMA model tuning.
script_3.py: File for Berkely Temperature Model building using Facebook's Prophet model
cru.py: File for CRU Temperature Model building using Facebook's Prophet model
grid_search.py: Grid Search for Prophet model.


## Libraries used:

1. Xarray
2. Matplotlib
3. Seaborn
4. Hvplot
5. Cartopy
6. Facebook Prophet
7. Pandas
8. statsmodels
