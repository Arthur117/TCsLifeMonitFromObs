from tqdm import tqdm
import numpy as np

### DATA UTILS

def create_dataset(ds_ibt, params_of_interest, PARAMS):
    '''
    ==> STRUCTURE OF X_crrent
    [usa_lon(t-4), usa_lon(t-3), usa_lon(t-2), ...., usa_rmw(t-3), usa_rmw(t-2), usa_rmw(t-1)] in the same way like params_of_interest
    e.g params_of_interest = ['usa_lon', 'usa_lat', 'usa_wind', 'usa_r34', 'usa_rmw']
    ==> STRUCTURE OF y
    [usa_lon(t), usa_lat(t), usa_wind(t), usa_r34(t), usa_rmw(t)]
    '''
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
        for i in range(len(ds.date_time) - PARAMS['seq_len']): # When it's negative, code doesn't enter the for loop
            ds_sub = ds.isel(date_time=slice(i, i + PARAMS['seq_len']))
            X_current = []
            y_current = []
            for param in params_of_interest:
                y_current.append(float(ds.isel(date_time=i + PARAMS['seq_len'])[param])) # target
                for e in ds_sub[param].values:
                    X_current.append(e) # inputs
            X.append(X_current)
            y.append(y_current)
    
    return X, y

def rmse(preds, targets):
    return np.sqrt(np.mean((preds - targets) ** 2))

def inverse_scale_normalize(X, MU, SIG, SCALE, param):
    return MU[param] + X * (SIG[param] / SCALE[param])