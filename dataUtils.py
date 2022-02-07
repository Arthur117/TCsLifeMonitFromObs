from tqdm import tqdm
import xarray as r
import numpy as np

### DATA UTILS

def create_dataset(ds_ibt, params_of_interest, PARAMS):
    X = [] # shape (n_samples, n_features)
    y = [] # shape (n_samples, n_targets)

    # For each storm, build X and y dataset.
    for s in tqdm(range(len(ds_ibt.storm))):
        # Select storm
        ds = ds_ibt.isel(storm=s)
        # Get only valid stime steps
        ds = ds[params_of_interest].mean(dim='quadrant', skipna=True)
        ds = ds.dropna(dim='date_time', subset=params_of_interest)

        # Add predictors and targets to the global dataset
        for i in range(len(ds.date_time) - PARAMS['time_steps_input']): # When it's negative, code doesn't enter the for loop
            ds_sub = ds.isel(date_time=slice(i, i + PARAMS['time_steps_input']))
            X_current = []
            y_current = []
            for param in params_of_interest:
                y_current.append(float(ds.isel(date_time=i + PARAMS['time_steps_input'])[param])) # target
                for e in ds_sub[param].values:
                    X_current.append(e) # inputs
            X.append(X_current)
            y.append(y_current)
    
    return X, y

def rmse(preds, targets):
    return np.sqrt(np.mean((preds - targets) ** 2))
