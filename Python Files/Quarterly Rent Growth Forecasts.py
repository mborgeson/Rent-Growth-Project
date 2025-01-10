###############################################################################################################
########################################## PYTHON PACKAGES TO IMPORT ##########################################
###############################################################################################################
import os
import sys
import time
import datetime
import enum
import winsound

import pandas as pd
import numpy as np
import scipy

import matplotlib
import matplotlib.pyplot as plt
import matplotlib_inline


import IPython
from IPython import get_ipython, paths
from IPython.paths import get_ipython_dir, get_ipython_module_path, get_ipython_package_dir
from IPython import display, extensions, extract_module_locals, Application
from IPython.display import HTML, SVG, YouTubeVideo, set_matplotlib_formats, FileLink, IFrame
from IPython.display import display, Pretty, DisplayHandle, DisplayObject
from IPython.core.pylabtools import figsize, getfigs
from IPython.core.interactiveshell import InteractiveShell
# from pylab import *


import nbformat
from nbformat import read
import notebook
from notebook import notebookapp, auth, serverextensions, extensions, utils
from notebook import nbextensions, nbconvert


import ipywidgets
from ipywidgets import AppLayout, FloatLogSlider, load_ipython_extension, widgets

import ipydatetime
import ipyparallel
import ipympl
import ipyleaflet

# import jupyterthemes
# from jupyterthemes import get_themes, install_theme, fonts, styles, layout
# import jupyterlab_sql
# from jupyterlab_sql import load_jupyter_server_extension


import statsmodels
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.stattools import durbin_watson
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.api import VAR
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.stats.outliers_influence import variance_inflation_factor


import sklearn
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor


import sktime
from sktime.datasets import load_airline, load_longley
from sktime.forecasting.base import ForecastingHorizon
# from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.theta import ThetaForecaster
# from sktime.forecasting.var import VAR
from sktime.utils.plotting import plot_series
from sktime.datatypes import get_examples
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error
from sktime.forecasting.fbprophet import Prophet


import torch
from torch import nn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


# import keras
# import tensorflow
# # Keras API for building and training models
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# # TensorFlow device management (to inspect hardware like GPUs/CPUs)
# from tensorflow.python.framework import config 
# from tensorflow.python.client import device_lib
# # TensorFlow build information (for debugging CUDA/cuDNN versions, etc.)
# from tensorflow.python.platform import build_info

import optuna

import openpyxl
import pyttsx3


start_time = time.time()



#####################################################################################
############################### EXCEL DATA FILE PATHS ###############################
#####################################################################################
user_name = "MattBorgeson"

fred_data_directory = os.path.abspath(r'C:\Users\MattBorgeson\OneDrive - B&R Capital\Programming Projects\Rent Growth Forecasting\Input Files\Fred Data')
fred_data_file_name_selection = os.listdir(fred_data_directory)[1]
fred_data_file_path = os.path.abspath(f'{fred_data_directory}\\{fred_data_file_name_selection}')

market_analysis_directory = os.path.abspath(r'C:\Users\MattBorgeson\OneDrive - B&R Capital\Programming Projects\Rent Growth Forecasting\Input Files\Market Analyses')
market_analysis_file_name_selection = os.listdir(market_analysis_directory)[1]
market_analysis_file_path = os.path.abspath(f'{market_analysis_directory }\\{market_analysis_file_name_selection}')



#####################################################################################
##################### READING EXISTING ANALYSES INTO DATAFRAMES #####################
#####################################################################################
os.listdir(fred_data_directory)
os.listdir(market_analysis_directory)

os.listdir(fred_data_directory)[1]
os.listdir(market_analysis_directory)[1]

market_analysis_df = pd.read_excel(market_analysis_file_path)
fred_data_df = pd.read_excel(fred_data_file_path)
market_analysis_df



#####################################################################################
######################## PYTORCH HARDWARE UTILIZATION CHECK #########################
#####################################################################################
print(torch.__version__)             # Prints the PyTorch version
print(torch.cuda.is_available())     # True if a compatible GPU is accessible


if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
else:
    print("CUDA is not available")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Create a random 2D tensor on the selected device
x = torch.rand((3, 3), device=device)
print("Tensor x:", x)
print("Tensor x is on:", x.device)




#######################################################################################################################
#################################### CREATE THE MERGED DATAFRAME & EXPORT TO EXCEL ####################################
#######################################################################################################################
market_analysis_df = pd.read_excel(market_analysis_file_path)
fred_data_df = pd.read_excel(fred_data_file_path)


merged_data_df = pd.merge(market_analysis_df, fred_data_df, left_on="Analysis Date", right_on="Adjusted Analysis Date", how='inner')
merged_data_df.rename(columns={"Analysis Date_x":"Analysis Date"}, inplace=True)
merged_data_df.rename(columns={"Adjusted Analysis Day":"Analysis Day"}, inplace=True)


# analysis_day_col = merged_data_df.pop("Analysis Day")
# merged_data_df.insert(3, "Analysis Day", analysis_day_col)

# merged_data_col_drop_list = [
#     "Analysis Date_y",
#     "Adjusted Analysis Date",
#     "Adjusted Analysis Year",
#     "Adjusted Analysis Month",
#     "Analysis Year",
#     "Analysis Month",
#     "Analysis Day",
#     "Annual Rent Growth (%)",
#     "Annual Rent Growth (%) - Lagged",
#     "Stabilized Vacancy (%)",
#     "Market Asking Rent/Unit ($)",
# ]

# merged_data_df.drop(columns=merged_data_col_drop_list, inplace=True)
# merged_data_df



merged_data_col_list = [
    "Analysis Date",
    "Market Name",

    # "Annual Rent Growth (%)",
    # "Annual Rent Growth (%) - Lagged",
    "Quarterly Rent Growth (%)",
    # "Quarterly Rent Growth (%) - Lagged",

    "Inventory (# Units)",
    "Under Construction (# Units)",
    "Under Construction (% of Inventory)",
    "Total Occupancy (# Units)",
    "Total Vacancy (# Units)",
    
    "Quarterly Net Deliveries (# Units)",
    "Quarterly Construction Starts (# Units)",
    "Quarterly Net Absorption (# Units)",
    "Absorption - Prior 12 Months (# Units)",
    "Net Deliveries - Prior 12 Months (# Units)",

    "Unemployment Rate (%)",
    "Job Growth (%)",
    "Population (#)",
    "Population Growth (%)",
    "Median Household Income ($)",
    "Median Household Income Growth (%)",

    "Federal Funds Effective Rate",

    "Consumer Price Index for All Urban Consumers: All Items in U.S. City Average",

    "Unemployed Persons in Phoenix-Mesa-Scottsdale, AZ (MSA)",
    "Employed Persons in Phoenix-Mesa-Scottsdale, AZ (MSA)",
    "Civilian Labor Force in Phoenix-Mesa-Scottsdale, AZ (MSA)",
    "Unemployment Rate in Phoenix-Mesa-Scottsdale, AZ (MSA)",

    "S&P CoreLogic Case-Shiller AZ-Phoenix Home Price Index",
]


quarterly_forecast_merged_data_df = merged_data_df[merged_data_col_list]

quarterly_forecast_merged_data_df = quarterly_forecast_merged_data_df[quarterly_forecast_merged_data_df != '-']
quarterly_forecast_merged_data_df["Quarterly Rent Growth (%)"] = quarterly_forecast_merged_data_df["Quarterly Rent Growth (%)"].astype('float')
quarterly_forecast_merged_data_df.dropna(inplace=True)
quarterly_forecast_merged_data_df.reset_index(inplace=True)
quarterly_forecast_merged_data_df.drop(columns=['index'], inplace=True)


############ Export the merged Dataframe to Excel
quarterly_forecast_merged_data_file_path = os.path.abspath(f'{fred_data_directory}\\Rent Growth Forecast (Quarterly) - Merged Data (Fred Series Data & Phoenix Market Data).xlsx')
quarterly_forecast_merged_data_df.to_excel(quarterly_forecast_merged_data_file_path, index=False)

merged_data_df = merged_data_df[merged_data_col_list]



#########################################################################################################
################################# ALL STATISTICAL CODE IN A SINGLE CELL #################################
#########################################################################################################
start_time = time.time()

# Load the dataset
merged_data_file_path = os.path.abspath(f'{fred_data_directory}\\Rent Growth Forecast (Quarterly) - Merged Data (Fred Series Data & Phoenix Market Data).xlsx')
merged_data = pd.read_excel(merged_data_file_path)
merged_data.dropna(inplace=True)

# Handle missing values with forward fill
df_filled = merged_data.fillna(method='ffill')


#################################### Add interaction terms ####################################
##### Interaction Type - Multiplication
df_filled['Job-Population Interaction'] = df_filled['Job Growth (%)'] * df_filled['Population Growth (%)']
df_filled['Income-Population Interaction'] = df_filled['Median Household Income Growth (%)'] * df_filled['Population Growth (%)']
df_filled['Job-Income Interaction'] = df_filled['Job Growth (%)'] * df_filled['Median Household Income Growth (%)']
df_filled['Job-Under Construction Interaction'] = df_filled['Job Growth (%)'] * df_filled['Under Construction (% of Inventory)']
df_filled['Population-Under Construction Interaction'] = df_filled['Population Growth (%)'] * df_filled['Under Construction (% of Inventory)']

##### Interaction Type - Division
# df_filled['Absorption-Deliveries Interaction'] = df_filled['Absorption - Prior 12 Months (# Units)'] / df_filled['Net Deliveries - Prior 12 Months (# Units)']
df_filled['Absorption-Deliveries Interaction'] = df_filled['Quarterly Net Absorption (# Units)'] / df_filled['Quarterly Net Deliveries (# Units)']

df_filled['Under Construction-Inventory Interaction'] = df_filled['Under Construction (# Units)'] / df_filled['Inventory (# Units)']
# df_filled['Deliveries-Inventory Interaction'] = df_filled['Net Deliveries - Prior 12 Months (# Units)'] / df_filled['Inventory (# Units)']
df_filled['Deliveries-Inventory Interaction'] = df_filled['Quarterly Net Deliveries (# Units)'] / df_filled['Inventory (# Units)']

df_filled['Occupancy-Inventory Interaction'] = df_filled['Total Occupancy (# Units)'] / df_filled['Inventory (# Units)']
df_filled['Vacancy-Inventory Interaction'] = df_filled['Total Vacancy (# Units)'] / df_filled['Inventory (# Units)']
df_filled['Unemployed-Labor Force Interaction'] = df_filled['Unemployed Persons in Phoenix-Mesa-Scottsdale, AZ (MSA)'] / df_filled['Civilian Labor Force in Phoenix-Mesa-Scottsdale, AZ (MSA)']
df_filled['Employed-Labor Force Interaction'] = df_filled['Employed Persons in Phoenix-Mesa-Scottsdale, AZ (MSA)'] / df_filled['Civilian Labor Force in Phoenix-Mesa-Scottsdale, AZ (MSA)']


# Add lagged variables for independent variables
df_filled['Unemployment Rate Lagged'] = df_filled['Unemployment Rate (%)'].shift(1)
df_filled['Job Growth Lagged'] = df_filled['Job Growth (%)'].shift(1)
df_filled['Population Growth Lagged'] = df_filled['Population Growth (%)'].shift(1)

# df_filled['Unemployment Rate Lagged'] = df_filled['Unemployment Rate (%)'].shift(4)
# df_filled['Job Growth Lagged'] = df_filled['Job Growth (%)'].shift(4)
# df_filled['Population Growth Lagged'] = df_filled['Population Growth (%)'].shift(4)


# Drop rows with NaN values caused by lagging
df_filled = df_filled.dropna()

# Define independent and dependent variables
X = df_filled[[
    'Job-Population Interaction',
    'Income-Population Interaction',
    'Job-Income Interaction',
    'Job-Under Construction Interaction',
    'Population-Under Construction Interaction',

    'Absorption-Deliveries Interaction',
    'Under Construction-Inventory Interaction',
    'Deliveries-Inventory Interaction',

    'Occupancy-Inventory Interaction',
    'Vacancy-Inventory Interaction',
    'Unemployed-Labor Force Interaction',
    'Employed-Labor Force Interaction',

    # 'Quarterly Rent Growth (%) - Lagged',
    'Unemployment Rate Lagged',
    'Job Growth Lagged',
    'Population Growth Lagged',

    # 'Inventory (# Units)',
    # 'Under Construction (# Units)',
    'Under Construction (% of Inventory)',
    # 'Absorption - Prior 12 Months (# Units)', 
    'Quarterly Net Absorption (# Units)',
    # 'Net Deliveries - Prior 12 Months (# Units)',
    'Quarterly Net Deliveries (# Units)',

    # 'Unemployment Rate (%)',
    # 'Job Growth (%)',
    # 'Population (#)',
    # 'Population Growth (%)', 
    'Median Household Income ($)',
    'Median Household Income Growth (%)',
               ]]

y = df_filled['Quarterly Rent Growth (%)']

# Add constant for the regression intercept
X = sm.add_constant(X)



##################################################################
# ------------------ 1. Out-of-Sample Testing ------------------ #
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Ridge regression with cross-validation hyperparameter tuning
ridge_params = {'alpha': [0.01, 0.1, 1.0, 10, 100]}
ridge_model = Ridge()
ridge_cv = GridSearchCV(ridge_model, ridge_params, cv=TimeSeriesSplit(n_splits=5), scoring='r2')
ridge_cv.fit(X_train, y_train)

# Best model from cross-validation
ridge_best = ridge_cv.best_estimator_

# Predictions on test set
ridge_predictions_train = ridge_best.predict(X_train)
ridge_predictions_test = ridge_best.predict(X_test)

# Ridge model performance
train_r2 = r2_score(y_train, ridge_predictions_train)
test_r2 = r2_score(y_test, ridge_predictions_test)

# Residual analysis
residuals = y_test - ridge_predictions_test
# print("Residuals summary:")
# print(residuals.describe())



########################################################################
# ------------------ 2. Vector Autoregression (VAR) ------------------ #
# Create VAR model using key time series
df_var = df_filled[[
    'Quarterly Rent Growth (%)',

    'Job-Population Interaction',
    'Income-Population Interaction',
    'Job-Income Interaction',
    'Job-Under Construction Interaction',
    'Population-Under Construction Interaction',

    'Absorption-Deliveries Interaction',
    'Under Construction-Inventory Interaction',
    'Deliveries-Inventory Interaction',

    'Occupancy-Inventory Interaction',
    'Vacancy-Inventory Interaction',
    'Unemployed-Labor Force Interaction',
    'Employed-Labor Force Interaction',

    'Unemployment Rate Lagged',
    'Job Growth Lagged',
    'Population Growth Lagged',

    'Under Construction (% of Inventory)',

    # 'Absorption - Prior 12 Months (# Units)', 
    'Quarterly Net Absorption (# Units)',
    # 'Net Deliveries - Prior 12 Months (# Units)',
    'Quarterly Net Deliveries (# Units)',

    'Unemployment Rate (%)',
    'Job Growth (%)',
    'Population Growth (%)', 
    'Median Household Income Growth (%)',
    ]]
model_var = VAR(df_var)
var_results = model_var.fit(maxlags=2)

# Forecast periods using VAR
# var_forecast_periods = 5
var_forecast_periods = 10
# var_forecast_periods = 20

var_forecast = var_results.forecast(df_var.values[-var_results.k_ar:], steps=var_forecast_periods)
var_forecast_df = pd.DataFrame(var_forecast, columns=df_var.columns)



##########################################################################
# ------------------ 3. ARIMA Time Series Forecasting ------------------ #
# Differencing to make the data stationary
df_filled['Quarterly Rent Growth Diff'] = df_filled['Quarterly Rent Growth (%)'].diff().dropna()

# ARIMA model (order based on dataset properties, p=1, d=1, q=1 as an example)
arima_model = ARIMA(df_filled['Quarterly Rent Growth (%)'].dropna(), order=(1, 1, 1))
arima_results = arima_model.fit()

# future_periods = 5
future_periods = 10
# future_periods = 20

arima_forecast = arima_results.forecast(steps=future_periods)


# Create future dates for the forecast
last_dataset_date = df_filled['Analysis Date'].max()
first_future_date = pd._libs.tslibs.timestamps.Timestamp(year=last_dataset_date.year, month=last_dataset_date.month + 1, day=last_dataset_date.day)

future_dates = pd.date_range(start=first_future_date, periods=future_periods, freq='QE', inclusive='right')


# Combine future dates and ARIMA forecast values
arima_forecast_df = pd.DataFrame({
    'Forecasted Date': future_dates,
    'ARIMA Forecasted Rent Growth (%)': arima_forecast
})




############################################################################
# ------------------ 4. Random Forest (Ensemble Method) ------------------ #
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

# Random forest performance
rf_r2 = r2_score(y_test, rf_predictions)



################################################################################
# ------------------ 5. LSTM Neural Network for Time Series ------------------ #

# Set environment variables for threads (similar idea to TensorFlow's environment vars)
os.environ["OMP_NUM_THREADS"] = "4"  # Controls OpenMP threads
os.environ["MKL_NUM_THREADS"] = "4"  # For MKL (if used internally)

# Then in your Python code:
torch.set_num_threads(4)  # Ensures PyTorch uses 2 threads internally


# Create a custom pytorch dataset and retun a dataloader
class TimeSeriesDataset(Dataset):
    """
    Custom PyTorch dataset for creating sequences of length `time_steps`
    from a 2D feature array X and 1D array y.
    """
    def __init__(self, X, y, time_steps):
        """
        X: numpy array of shape [num_samples, num_features]
        y: numpy array of shape [num_samples]
        time_steps: int, number of past timesteps to include in each sample
        """
        self.X = X
        self.y = y
        self.time_steps = time_steps

    def __len__(self):
        # Number of valid sequences = total samples - time_steps
        return len(self.X) - self.time_steps

    def __getitem__(self, idx):
        # Sequences range from idx to idx + time_steps
        x_seq = self.X[idx : idx + self.time_steps]
        y_label = self.y[idx + self.time_steps]  # label at the end of the window
        # Convert to float32 tensors
        x_seq = torch.tensor(x_seq, dtype=torch.float32)
        y_label = torch.tensor(y_label, dtype=torch.float32)
        return x_seq, y_label


def create_torch_dataloader(X, y, time_steps, batch_size):
    """
    Create a PyTorch DataLoader for time series data.
    """
    dataset = TimeSeriesDataset(X, y, time_steps)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,   # Typically don't shuffle time series
        drop_last=False  # Keep all samples
    )
    return dataloader


# Scaling Data
# Example: X, y are your original time series data
# They can be NumPy arrays or Pandas DataFrame/Series.

scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))

X_scaled = scaler_X.fit_transform(X)  # shape: [samples, features]
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()


# Define the LSTM model
class LSTMTimeSeriesModel(nn.Module):
    def __init__(self, input_dim, lstm_units):
        """
        input_dim : number of features per timestep
        lstm_units: hidden dimension (number of LSTM units)
        """
        super(LSTMTimeSeriesModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size=input_dim, hidden_size=lstm_units, batch_first=True, dropout=0.0)
        self.lstm2 = nn.LSTM(input_size=lstm_units, hidden_size=lstm_units, batch_first=True, dropout=0.0)
        self.fc = nn.Linear(lstm_units, 1)  # Final dense layer -> 1 output

    def forward(self, x):
        # x shape: [batch_size, time_steps, input_dim]
        out, (h, c) = self.lstm1(x)  # out shape: [batch_size, time_steps, lstm_units]
        out, (h, c) = self.lstm2(out)  # out shape: [batch_size, time_steps, lstm_units]
        # Take the last timestep's output
        last_timestep = out[:, -1, :]  # shape: [batch_size, lstm_units]
        return self.fc(last_timestep)  # shape: [batch_size, 1]


# Train and evaluate the mnodel using the dataloader
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    
    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs.view(-1), y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x_batch.size(0)
    
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def evaluate(model, dataloader, criterion, device):
    model.eval()
    eval_loss = 0.0
    
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = model(x_batch)
            loss = criterion(outputs.view(-1), y_batch)
            eval_loss += loss.item() * x_batch.size(0)

    eval_loss = eval_loss / len(dataloader.dataset)
    return eval_loss








#############################################################################
# -------- 6. Forecasting/Prediction with the Best Hyperparameters -------- #
# Forecasting
time_steps = 10
num_epochs = 50


def run_optuna_study(X_scaled, y_scaled, time_steps, num_epochs, n_splits=5, n_trials=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tscv = TimeSeriesSplit(n_splits=n_splits)

    def objective(trial):
        # lstm_units = trial.suggest_int('lstm_units', 16, 64)
        lstm_units = trial.suggest_int('lstm_units', 16, 128)        

        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)

        # batch_size = trial.suggest_categorical('batch_size', [16, 32])
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])

        avg_loss = 0.0
        fold_count = 0

        for train_idx, test_idx in tscv.split(X_scaled):
            # Split data
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y_scaled[train_idx], y_scaled[test_idx]
            
            # Create Dataloaders
            train_loader = create_torch_dataloader(X_train, y_train, time_steps, batch_size)
            test_loader  = create_torch_dataloader(X_test,  y_test,  time_steps, batch_size)
            
            # Build model
            input_dim = X_train.shape[1]
            model = LSTMTimeSeriesModel(input_dim=input_dim, lstm_units=lstm_units).to(device)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            
            # Train model for num_epochs
            for epoch in range(num_epochs):
                train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            
            # Evaluate on test set
            test_loss = evaluate(model, test_loader, criterion, device)
            avg_loss += test_loss
            fold_count += 1
        
        return avg_loss / fold_count

    # Create and run the study
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, n_jobs=-1)  # Parallel or single-thread

    return study.best_params


best_hyperparameters = run_optuna_study(X_scaled, y_scaled, time_steps, num_epochs, n_splits=5, n_trials=10)
print("Best Hyperparams:", best_hyperparameters)


# Train the model using the best hyperparameters
# Suppose you got these from `best_hyperparameters`
num_lstm_units = best_hyperparameters['lstm_units']
learning_rate  = best_hyperparameters['learning_rate']
batch_size     = best_hyperparameters['batch_size']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create a single DataLoader over the entire dataset
full_loader = create_torch_dataloader(X_scaled, y_scaled, time_steps, batch_size)

model_lstm = LSTMTimeSeriesModel(
    input_dim=X_scaled.shape[1], 
    lstm_units=num_lstm_units
).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model_lstm.parameters(), lr=learning_rate)

# num_epochs = 20
for epoch in range(num_epochs):
    train_loss = train_one_epoch(model_lstm, full_loader, criterion, optimizer, device)
    # If desired, implement an early stopping check here
    # ...
    # print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}")

# Optionally save your final model weights
checkpoint_filepath = os.path.abspath(r"C:\Users\MattBorgeson\OneDrive - B&R Capital\Programming Projects\Rent Growth Forecasting\Output Files\Best Model Weights/best_model_fold_torch (Quarterly).pt")
# checkpoint_filepath = os.path.abspath("best_model_fold_torch.pt")
torch.save(model_lstm.state_dict(), checkpoint_filepath)

# To load them later:
# model_lstm.load_state_dict(torch.load(checkpoint_filepath))
# model_lstm.eval()


# Autoregressive Forecasting with LSTM
model_lstm.eval()

last_sequence = X_scaled[-time_steps:].copy()  # shape: (time_steps, num_features)
ml_optimized_predictions = []


forecast_num_steps = 10  # e.g., 10 future steps
for _ in range(forecast_num_steps):
    # (1) Reshape for model: [batch_size=1, time_steps, num_features]
    seq_input = torch.tensor(last_sequence, dtype=torch.float32).unsqueeze(0).to(device)
    
    # (2) Predict
    with torch.no_grad():
        next_value_scaled_tensor = model_lstm(seq_input)  # shape: [1, 1]
    
    next_value_scaled = next_value_scaled_tensor.cpu().numpy().reshape(-1, 1)
    
    # (3) Inverse transform from scaled to original
    next_value_original_scale = scaler_y.inverse_transform(next_value_scaled)
    ml_optimized_predictions.append(next_value_original_scale[0, 0])
    
    # (4) Update the last_sequence with this new predicted value
    #     If your target is the 1st column, place it there:
    next_features = np.zeros((1, last_sequence.shape[1]))
    next_value_rescaled = scaler_y.transform(next_value_original_scale)  # scale back for model continuity
    next_features[0, 0] = next_value_rescaled[0, 0]
    
    # Shift and append
    last_sequence = np.vstack([last_sequence[1:], next_features])

ml_optimized_forecast_df = arima_forecast_df.copy(deep=True)
ml_optimized_forecast_df["Machine Learning Optimized Forecast"] = ml_optimized_predictions
ml_optimized_forecast_df.drop(columns=['ARIMA Forecasted Rent Growth (%)'], inplace=True, errors='ignore')
ml_optimized_forecast_df.reset_index(inplace=True, drop=True)




##############################################################
# ------------------ 7. Summary & Outputs ------------------ #
rent_growth_forecasts_df = arima_forecast_df.copy(deep=True)
rent_growth_forecasts_df["Machine Learning Optimized Forecast"] = list(ml_optimized_forecast_df["Machine Learning Optimized Forecast"])
rent_growth_forecasts_df['Vector Autoregression (VAR) Forecast'] = list(var_forecast_df['Quarterly Rent Growth (%)'])
rent_growth_forecasts_df.rename(columns={'ARIMA Forecasted Rent Growth (%)':'ARIMA Forecast'}, inplace=True)


# print(f"Ridge Regression R^2 on Train: {train_r2}, Test: {test_r2}")
# print(f"Random Forest R^2 on Test: {rf_r2}")


var_forecast_df.rename(columns={'Quarterly Rent Growth (%)':'Vector Autoregression (VAR) Forecast'}, inplace=True)
print("Forecast Type -- Vector Autoregression Analysis")
print("Forecast Tools -- Statsmodels")
print(f"VAR Forecast:")
print(var_forecast_df['Vector Autoregression (VAR) Forecast'])


print("Forecast Type -- Autoregressive Integrated Moving Average (ARIMA) Analysis")
print("Forecast Tools -- Statsmodels")
print(f"ARIMA Forecast:")
print(arima_forecast_df['ARIMA Forecasted Rent Growth (%)'])


print("Forecast Type -- Machine Learning/Neural Network Forecasting Using Optimized Independent Variable/Parameter Weights")
print("Forecast Tools -- PyTorch & Optuna")
# print(f"Forecasted values for the next {forecast_num_steps} time steps:")
print(f"Machine Learning Optimized Forecast:")
print(ml_optimized_forecast_df["Machine Learning Optimized Forecast"])


# Save the Forecasts DataFrame with all forecasts to an Excel file
forecasting_analysis_outputs_file_path = os.path.abspath(r'C:\Users\MattBorgeson\OneDrive - B&R Capital\Programming Projects\Rent Growth Forecasting\Output Files\Forecasts')


var_forecast_file_path = os.path.abspath(f'{forecasting_analysis_outputs_file_path}\\VAR Forecast Model Forecasts (Quarterly).xlsx')
arima_forecast_file_path = os.path.abspath(f'{forecasting_analysis_outputs_file_path}\\ARIMA Forecast Model Forecasts (Quarterly).xlsx')
ml_optimized_forecast_file_path = os.path.abspath(f'{forecasting_analysis_outputs_file_path}\\Machine Learning Forecast Model Forecasts (Quarterly).xlsx')
all_forecasts_file_path = os.path.abspath(f'{forecasting_analysis_outputs_file_path}\\All Forecast Methodologies Forecasts (Quarterly).xlsx')


# var_forecast_df.to_excel(var_forecast_file_path, index=False)
# arima_forecast_df.to_excel(arima_forecast_file_path, index=False)
# ml_optimized_forecast_df.to_excel(ml_optimized_forecast_file_path, index=False)
rent_growth_forecasts_df.to_excel(all_forecasts_file_path, index=False)



##############################################################
# -------------------- 8. Code Run Time -------------------- #
end_time = time.time()
elapsed_time = end_time - start_time
elapsed_time = (elapsed_time/60)
elapsed_time = round(elapsed_time, 2)
print(f"Elapsed time: {elapsed_time} minutes")

def code_completion_notification ():
    engine = pyttsx3.init()
    # engine.endLoop()    
    
    try:
        engine.endLoop()
        del engine
        engine = pyttsx3.init()
    except:
        pass
        code_run_time = f"The code took {elapsed_time} minutes to run."
        text = "Quarterly Rent Growth Forecast Analysis Complete."
        text = f"{text} {code_run_time}"

        engine.setProperty('rate', 225)

        volume = engine.getProperty('volume')
        engine.setProperty('volume', volume+0.50)

        winsound.Beep(750, 1250)
        engine.say(text)

        engine.startLoop(False)
        engine.iterate()
        engine.endLoop()        


code_completion_notification ()
# %run "{python_files_folder}Quarterly Rent Growth Forecasts.py"
