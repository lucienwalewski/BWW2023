# Code adapted from EY Challenge Notebook Level 2 Notebook V3.0 

# Visualization
import ipyleaflet
import matplotlib.pyplot as plt
from IPython.display import Image
import seaborn as sns

# Data Science
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Feature Engineering
from sklearn.model_selection import train_test_split

# Machine Learning
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score


# Planetary Computer Tools
import pystac
import pystac_client
import odc
from pystac_client import Client
from pystac.extensions.eo import EOExtension as eo
from odc.stac import stac_load
import planetary_computer as pc

# Please pass your API key here
pc.settings.set_subscription_key('5f3a374adb2b43fe89373be784ae30c5')

# Others
import requests
import rich.table
from itertools import cycle
from tqdm import tqdm


# Global 
# Define the pixel resolution for the final product
# Define the scale according to our selected crs, so we will use degrees

RESOLUTION = 10  # meters per pixel 
SCALE = RESOLUTION / 111320.0 # degrees per pixel for crs=4326

# with extra month before and after 
time_of_interest_WS = "2021-12-01/2022-05-30"
time_of_interest_SA = "2022-04-01/2022-09-30"


def access_sentinel_data(longitude, 
                         latitude, 
                         season, 
                         assests=["vv", "vh"], 
                         box_size_deg=0.0006):
    '''
    Returns data to compute indices 
    Inputs are lat, long, season, assests, surrounding box in degrees
    '''

    bands_of_interest = assests
    if season == 'SA':
        time_slice = "2022-05-01/2022-08-31"
    if season == 'WS':
        time_slice = "2022-01-01/2022-04-30"
        
    vv_list = []
    vh_list = []
    vv_by_vh_list = []

    min_lon = longitude-box_size_deg/2
    min_lat = latitude-box_size_deg/2
    max_lon = longitude+box_size_deg/2
    max_lat = latitude+box_size_deg/2
    
    bbox = (min_lon, min_lat, max_lon, max_lat)
    time_of_interest = time_slice
    
    catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    search = catalog.search(collections=["sentinel-1-rtc"], bbox=bbox, datetime=time_of_interest)
    items = list(search.get_all_items())

    data = stac_load(items,bands=assests, 
                     patch_url=pc.sign, 
                     bbox=bbox, 
                     crs="EPSG:4326", 
                     resolution=SCALE)
    
    mean = data.mean(dim=['latitude','longitude']).compute()
              
    return mean

def compute_rvi(mean):
    '''
    Input sentinel data for a lat-long region of interest 
    Output the rvi phenology stats 
    '''

    dop = (mean.vv / (mean.vv + mean.vh))
    m = 1 - dop
    rvi = (np.sqrt(dop))*((4*mean.vh)/(mean.vv + mean.vh))

    resample_period = "2W"
    window = 4

    rvi_ts = pd.Series(rvi, index=rvi.time)

    rvi_smooth = (
        rvi_ts.resample(resample_period)
        .median()
        .rolling(window, min_periods=1)
        .mean()
    )

    return rvi_smooth

def compute_rvi_stats(rvi_smooth, harvest):
    # get two mins (end and start)
    rvi_handling = rvi_smooth.copy()
    down = rvi_handling.diff().values < 0
    rvi_handling['mins'] = np.r_[down[1:] != down[:-1], False] & down

    local_mins = np.where(rvi_handling['mins']==True)

    start_index = local_mins[0][0]
    end_index = local_mins[0][-1]

    start_val = rvi_handling[end_index]
    end_val = rvi_handling[end_index]

    start_date = (rvi_handling.index[end_index] - rvi_handling.index[0]).days
    end_date = (rvi_handling.index[end_index] - rvi_handling.index[0]).days

    # when not enough local mins... 
    if end_date < 90:
        end_date = 150
        end_index = rvi_handling['mins'].shape[0]-1
        end_val = rvi_handling[end_index]
    if start_date > 90:
        start_date = 30
        start_index = 0
        start_val = rvi_handling[start_index]

    max_val = rvi_smooth[start_index:end_index+1].max()
    max_index = rvi_smooth[start_index:end_index+1].idxmax()
    max_date = (max_index - rvi_handling.index[0]).days

    harvest_date = (pd.to_datetime(harvest)- rvi_handling.index[0]).days
    date_diff = [(pd.to_datetime(harvest)- rvi_handling.index[i]).days for i in range(rvi_smooth.values.shape[0])]
    date_diff = np.array(date_diff)
    goes_neg = np.where(date_diff<=0)[0][0]
    if date_diff[goes_neg] == 0:
        harvest_val = rvi_smooth.index[goes_neg]

    else:
        days_neg = [(rvi_handling.index[goes_neg-1] - rvi_handling.index[0]).days, 
                    (rvi_handling.index[goes_neg] - rvi_handling.index[0]).days]
        print(days_neg)
        vals_neg = [rvi_smooth.values[goes_neg-1], rvi_smooth.values[goes_neg]]
        print(vals_neg)
        m = (vals_neg[1] - vals_neg[0])/(days_neg[1]-days_neg[0])
        
        harvest_val = m*(harvest_date - days_neg[0]) + vals_neg[0]

    rvi_correlation = sm.tsa.acf(rvi_smooth)[1]

    return [start_date, end_date, 
            start_val, end_val, 
            max_date, max_val, 
            harvest_date, harvest_val, 
            rvi_correlation]


def get_all_features(crop_yield_data):
    all_features = []
    debug_stop = 0
    for ind, row in crop_yield_data.iterrows():
        if debug_stop > 2:
            break
        debug_stop+=1
        mean_row = access_sentinel_data(row["Longitude"], 
                                        row["Latitude"], 
                                        row["Season(SA = Summer Autumn, WS = Winter Spring)"])
        mean_row = compute_rvi(mean_row)
        all_features.append(compute_rvi_stats(mean_row, row["Date of Harvest"]))
    return all_features, debug_stop


def build_model(X, y):
    # Choose any random state
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=21)
    regressor = ExtraTreesRegressor(bootstrap=False, ccp_alpha=0.0, criterion='mse',
                    max_depth=None, max_features='auto', max_leaf_nodes=None,
                    max_samples=None, min_impurity_decrease=0.0, min_samples_leaf=1,
                    min_samples_split=2, min_weight_fraction_leaf=0.0,
                    n_estimators=100, n_jobs=-1, oob_score=False,
                    random_state=123, verbose=0, warm_start=False)
    regressor.fit(X_train, y_train)
    insample_predictions = regressor.predict(X_train)
    outsample_predictions = regressor.predict(X_test)
    r2_insample = r2_score(y_train,insample_predictions)
    r2_outsample = r2_score(y_test,outsample_predictions)

    return r2_insample, r2_outsample







