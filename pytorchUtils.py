# from tqdm import tqdm
import xarray as xr
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

### PYTORCH UTILS

class LSTM1(nn.Module):
    '''
    LSTM for 02_lstm_dynamic_pred_ibtracsv01.ipynb and _ibtracsv01.ipynb
    1 timestamp only, and n_features = 20.
    '''
    def __init__(self, num_classes=5, input_size=5, hidden_size=2, num_layers=1, seq_len=4, dropout=0):
        super(LSTM1, self).__init__()
        self.num_classes = num_classes # number of classes
        self.num_layers  = num_layers  # number of layers
        self.input_size  = input_size  # input size
        self.hidden_size = hidden_size # hidden state
        self.seq_len     = seq_len     # sequence length
        self.dropout     = dropout

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=dropout) # lstm
        self.fc_1 =  nn.Linear(hidden_size, 128)                                      # fully connected 1
        self.fc   = nn.Linear(128, num_classes)                                       # fully connected last layer

        self.relu = nn.ReLU()
    
    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(x.device.type) # hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(x.device.type) # internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) # lstm with input, hidden, and internal state
        hn  = hn[self.num_layers - 1, :, :]
        hn  = hn.view(-1, self.hidden_size) # reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc_1(out) # first Dense
        out = self.relu(out) # relu
        out = self.fc(out)   # final Output
        return out
    
class MLP(nn.Module):
    '''
    Multilayer Perceptron.
    '''
    def __init__(self, input_size=5, seq_len=4, output_size=1):
        super().__init__()
        self.input_size  = input_size  # input size
        self.seq_len     = seq_len     # sequence length
        self.output_size = output_size # output size
        
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.input_size * self.seq_len, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
            )
        
    def forward(self, x):
        return self.layers(x)

    
class LinearRegression(nn.Module):
    '''
    Linear Regression.
    '''
    def __init__(self, input_size=5, seq_len=4, output_size=1):
        super().__init__()
        self.input_size  = input_size  # input size
        self.seq_len     = seq_len     # sequence length
        self.output_size = output_size # output size
        
        self.lin = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.input_size * self.seq_len, self.output_size)
        )
        
    def forward(self, x):
        return self.lin(x)
    
class ShortTimeseriesDataset(Dataset):   
    '''
    Short Timeseries Dataset. 
    Serves as input to DataLoader to transform X 
      into sequence data using rolling window. 
    DataLoader using this dataset will output batches 
      of `(batch_size, seq_len, n_features)` shape.
    Suitable as an input to RNNs or LSTMs. 
    '''
    def __init__(self, X: xr.Dataset, y: np.ndarray, n_features: int=5, seq_len: int=4, device: str='cpu'):
        self.X          = torch.tensor(X).float()
        self.y          = torch.tensor(y).float()
        self.n_features = n_features
        self.seq_len    = seq_len
        self.device     = device

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        # Select data
        input_tensor  = torch.transpose(self.X[i, :].reshape(self.n_features, self.seq_len), 0, 1)
        target_tensor = self.y[i]
        # Move to GPU if available
        input_tensor  = input_tensor.to(self.device)
        target_tensor = target_tensor.to(self.device)
        return input_tensor, target_tensor
    
class CompleteTimeseriesDataset(Dataset):   
    '''
    Complete Timeseries Dataset.
    Checks that usa_rmw is contains at least a 5-days sequence of non NaN values.
    In that case, fills the NaN values with 0 padding, for every variable.
    '''
    def __init__(self, X: np.ndarray, y: np.ndarray,  device: str='cpu'):
        self.X          = torch.tensor(X).float()
        self.y          = torch.tensor(y).float()
        self.device     = device

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        # Select data & Move to GPU if available
        input_tensor  = self.X[i, :, :].to(self.device)
        target_tensor = self.y[i, 0, :].to(self.device)
        return input_tensor, target_tensor

    
    
    
###

class ShortTermPredDataModule(pl.LightningDataModule):
    '''
    PyTorch Lighting DataModule subclass.
    https://www.kaggle.com/tartakovsky/pytorch-lightning-lstm-timeseries-clean-code
    
    Serves the purpose of aggregating all data loading 
      and processing work in one place.
    '''
    
    def __init__(self, PATHS={'ibtracs_data': '/home/arthur/data/ibtracs/IBTrACS.NA.v04r00.nc'}, seq_len = 4, batch_size = 32, num_workers=0):
        super().__init__()
        self.PATHS         = PATHS
        self.seq_len       = seq_len
        self.batch_size    = batch_size
        self.num_workers   = num_workers
        self.X_train       = None
        self.y_train       = None
        self.X_val         = None
        self.y_val         = None
        self.X_test        = None
        self.X_test        = None
        self.columns       = None
        self.preprocessing = None

    def setup(self, stage=None):
        '''
        Data is resampled to hourly intervals.
        Both 'np.nan' and '?' are converted to 'np.nan'
        'Date' and 'Time' columns are merged into 'dt' index
        '''

        if stage == 'fit' and self.X_train is not None:
            return 
        if stage == 'test' and self.X_test is not None:
            return
        if stage is None and self.X_train is not None and self.X_test is not None:  
            return
        
        path = '/kaggle/input/electric-power-consumption-data-set/household_power_consumption.txt'
        
        df = pd.read_csv(
            path, 
            sep=';', 
            parse_dates={'dt' : ['Date', 'Time']}, 
            infer_datetime_format=True, 
            low_memory=False, 
            na_values=['nan','?'], 
            index_col='dt'
        )

        df_resample = df.resample('h').mean()

        X = df_resample.dropna().copy()
        y = X['Global_active_power'].shift(-1).ffill()
        self.columns = X.columns


        X_cv, X_test, y_cv, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
    
        X_train, X_val, y_train, y_val = train_test_split(
            X_cv, y_cv, test_size=0.25, shuffle=False
        )

        preprocessing = StandardScaler()
        preprocessing.fit(X_train)

        if stage == 'fit' or stage is None:
            self.X_train = preprocessing.transform(X_train)
            self.y_train = y_train.values.reshape((-1, 1))
            self.X_val = preprocessing.transform(X_val)
            self.y_val = y_val.values.reshape((-1, 1))

        if stage == 'test' or stage is None:
            self.X_test = preprocessing.transform(X_test)
            self.y_test = y_test.values.reshape((-1, 1))
        

    def train_dataloader(self):
        train_dataset = TimeseriesDataset(self.X_train, 
                                          self.y_train, 
                                          seq_len=self.seq_len)
        train_loader = DataLoader(train_dataset, 
                                  batch_size = self.batch_size, 
                                  shuffle = False, 
                                  num_workers = self.num_workers)
        
        return train_loader

    def val_dataloader(self):
        val_dataset = TimeseriesDataset(self.X_val, 
                                        self.y_val, 
                                        seq_len=self.seq_len)
        val_loader = DataLoader(val_dataset, 
                                batch_size = self.batch_size, 
                                shuffle = False, 
                                num_workers = self.num_workers)

        return val_loader

    def test_dataloader(self):
        test_dataset = TimeseriesDataset(self.X_test, 
                                         self.y_test, 
                                         seq_len=self.seq_len)
        test_loader = DataLoader(test_dataset, 
                                 batch_size = self.batch_size, 
                                 shuffle = False, 
                                 num_workers = self.num_workers)

        return test_loader    
    
    
    
    