from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import copy

### FUNCTIONS

def coriolis(lat):
    '''Latitude must be in degrees.'''
    Omega = 7.2921e-5                             # Earth rotation vector
    f     = 2 * Omega * np.sin(lat * np.pi / 180) # Coriolis parameter at lat° latitude and assuming it's constant 
    return f

def get_rmax_ck22(Vmax, R17, fcor, intercept, coef1, coef2):
    M17     = R17 * 1000 * 17.5 + 0.5 * fcor * ((R17 * 1000) ** 2) 
    M_ratio = intercept * np.exp(coef1 * (Vmax - 17.5) + coef2 * (Vmax - 17.5) * 0.5 * fcor * R17 * 1000)
    Mmax    = M_ratio * M17
    Rmax    = (Vmax / fcor) * (np.sqrt(1 + (2 * fcor * Mmax) / (Vmax ** 2)) - 1)
    return Rmax

def create_Xt_1_and_Xt(ds_ibt, params_of_interest, final_params=['usa_wind', 'usa_rmw', 'usa_r34', 'fcor', 'u_trans', 'v_trans'], fcor_boost=1):
    '''So far, fcor is boosted by 1e6'''
    Xt_1 = [] # shape (n_samples, n_features)
    Xt   = [] # shape (n_samples, n_targets)
    
    # For each storm, build Xt_1 and Xt dataset.
    for s in tqdm(range(len(ds_ibt.storm))):
        # Select storm
        ds = ds_ibt.isel(storm=s)
        # Add derivatives
        for p in ['usa_rmw', 'usa_r34', 'usa_wind']:
            ds['{}_diff'.format(p)]     = ds[p] * np.nan
            ds['{}_diff'.format(p)][1:] = ds[p].diff(dim='date_time')
        # Add Coriolis
        ds['fcor'] = coriolis(np.abs(ds['usa_lat'])) * fcor_boost
        # Get only valid stime steps
        ds         = ds.dropna(dim='date_time', subset=params_of_interest + ['usa_rmw_diff', 'usa_r34_diff', 'usa_wind_diff'])

        # Add to X and Y dataset
        # final_params = ['usa_wind', 'usa_wind_diff', 'usa_rmw', 'usa_rmw_diff', 'usa_r34', 'usa_r34_diff', 'fcor']
        da           = ds[final_params].to_array().transpose()
        # print(da)
        for t in range(len(da['date_time']) - 1):
            Xt_1.append(da[t, :].values)
            Xt.append(da[t + 1, :].values)
        # print(Xt)
    
    # Convert to arrays
    Xt_1 = np.array(Xt_1)
    Xt   = np.array(Xt)
    
    return Xt, Xt_1

def create_Xt_1_and_Xt_full(ds_ibt, final_params=['usa_wind', 'usa_rmw', 'rmax_ck22', 'usa_r34', 'fcor', 'u_trans', 'v_trans']):
    print('Creating dataset...')
    Xt_1 = [] # shape (n_samples, n_features)
    Xt   = [] # shape (n_samples, n_targets)
    fin_par_with_diff = final_params + ['{}_diff'.format(p) for p in final_params]
    
    # For each storm, build Xt_1 and Xt dataset.
    for s in tqdm(range(len(ds_ibt.storm))):
        # Select storm
        ds = ds_ibt.isel(storm=s)
        # Add derivatives
        for p in final_params:
            ds['{}_diff'.format(p)]     = ds[p] * np.nan
            ds['{}_diff'.format(p)][1:] = ds[p].diff(dim='date_time')
        # Get only valid stime steps
        ds         = ds.dropna(dim='date_time', subset=fin_par_with_diff)

        # Add to X and Y dataset
        da         = ds[fin_par_with_diff].to_array().transpose()
        for t in range(len(da['date_time']) - 1):
            Xt_1.append(da[t, :].values)
            Xt.append(da[t + 1, :].values)
    
    # Convert to arrays
    Xt_1 = np.array(Xt_1)
    Xt   = np.array(Xt)
    
    # Print
    print('Shape of Xt = {}'.format(Xt.shape))
    print('Final Parameters = {}'.format(fin_par_with_diff))
    
    return Xt, Xt_1

def get_loglikelihood_vmax(param, x_a, P_a, Y):
    """Given a Kalman Filter param and registered analyzed states x_a and P_a, and observations, 
    computes the innovation log-likelihood at each time step, but only on the Vmax component. 
    Returns np.nan when observation are masked.    
    """
    x_f          = []
    P_f          = []
    Log_lik_vmax = []

    x_f.append(param.initial_state_mean)
    P_f.append(param.initial_state_covariance)
    for t in range(Y.shape[0] - 1):
        x_f.append(np.dot(param.transition_matrices, x_a[t]))
        P_f.append(np.dot(np.dot(param.transition_matrices, P_a[t]), param.transition_matrices.T) + param.transition_covariance)
        A   = copy.deepcopy((Y[t, :] - np.dot(param.observation_matrices, x_f[t])))
        SIG = np.dot(np.dot(param.observation_matrices, P_f[t]), param.observation_matrices.T) + param.observation_covariance
        B   = copy.deepcopy(np.linalg.inv(SIG))
        # Mask all values that don't relate to Vmax
        A[1:] = 0
        try: 
            Log_lik_vmax.append(0.5 * A.T@(B@A) - 0.5 * (np.log(2 * np.pi) + SIG[0, 0]))
        except ValueError: # When the value is masked, we add 0
            Log_lik_vmax.append(np.nan)
        
    return Log_lik_vmax, np.array(x_f), np.array(P_f)

def rmse(X, Y):
    return np.sqrt(np.mean((X - Y) ** 2))

def em_optimization(state_space_model, Y, n_iter=10):
    loglikelihoods    = np.zeros(n_iter)
    for i in range(len(loglikelihoods)):
        '''Bug solved by changing standard.py
        Details at https://stackoverflow.com/questions/37730850/when-using-pykalman-python-kalman-filter-what-data-type-does-loglikelihood-fu'''
        loglikelihoods[i] = state_space_model.loglikelihood(Y)
        state_space_model.em(Y, n_iter=1)
    return state_space_model, loglikelihoods

