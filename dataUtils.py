from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

### DATA UTILS

def plot_hist(da, bins=10, density=False):
    '''Given a xr.DataArray as input, plot the histogram of its values.'''
    plt.title(da.name, weight='bold')
    plt.hist(np.array(da).flatten(), bins=bins, density=density);plt.grid();plt.show()
    return None

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

def get_X_and_y_arrays_from_Dataset(ds_ibt, input_variables, target_variable):
    '''
    Given a xarray.Dataset ds_ibt of storms and sequences (e.g dim= date_time) of variables of interest,
    returns an input X and a target y in np.array format.
    These can be passed as input of the CompleteTimeseriesDataset(Dataset) class.
    '''
    X = np.array(list(ds_ibt[input_variables].data_vars.values()))
    X = np.reshape(X, (X.shape[1], X.shape[0], X.shape[2])) # shape (n_storms, n_features, date_time) = (115, 4, 360)
    y = np.array(list(ds_ibt[target_variable].data_vars.values()))
    y = np.reshape(y, (y.shape[1], y.shape[0], y.shape[2])) # shape (n_storms, n_features, date_time) = (115, 1, 360)
    return X, y

def rmse(preds, targets):
    return np.sqrt(np.mean((preds - targets) ** 2))

def inverse_scale_normalize(X, MU, SIG, SCALE, param):
    return MU[param] + X * (SIG[param] / SCALE[param])

def save_ibt_sample(ds, path, params_of_interest=['usa_lon', 'usa_lat', 'usa_wind', 'usa_r34', 'usa_rmw']):
    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(25, 25))

    for i, ax in enumerate(fig.axes[:-1]):
        param = params_of_interest[i]
        ax.set_title(param, weight='bold')
        ax.plot(ds[param]);ax.grid()

    ax  = fig.axes[-1]
    if ds['name'].values.shape:
        idx = np.where(~pd.isnull(ds['name'].values))[0][0]
        ax.text(0.02, 0.95, str(ds['name'].values[idx])[2:-1], weight='bold')
        idx = np.where(~pd.isnull(ds['sid'].values))[0][0]
        ax.text(0.02, 0.88, 'SID  = %s'%str(ds['sid'].values[idx])[2:-1])
        ax.text(0.02, 0.81, 'YEAR = %s'%(str(ds['time'].values[0])[:4]))
    else:
        ax.text(0.02, 0.95, str(ds['name'].values)[2:-1], weight='bold')
        ax.text(0.02, 0.88, 'SID  = %s'%str(ds['sid'].values)[2:-1])
        ax.text(0.02, 0.81, 'YEAR = %s'%(str(ds['time'].values[0])[:4]))
    
    plt.savefig(path)
    plt.clf()

