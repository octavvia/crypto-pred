# !pip install arch
# !pip3 install yfinance

# Librarys of visualization
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date
from scipy import stats
import streamlit as st
import datetime

# %matplotlib inline

# Turn off all the warnings and messages
import warnings
warnings.filterwarnings('ignore')

# Getting financial data
import yfinance as yf
# Indexation of time
from datetime import datetime, timedelta
# The measures of performances
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
# Libraries of Autocorrelation and Partial  Autocorrelation for identifier lag of GARCH model
import statsmodels.api as sm
import statsmodels.graphics.tsaplots as sgt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, q_stat, adfuller
from statsmodels.tsa.stattools import adfuller

# GARCH modeling
from arch import arch_model
from numpy import cumsum, log, polyfit, sqrt, std, subtract
from scipy.stats import probplot, moment

# To see all of the text, change the column width in pandas
# (Model performance will be shown later)
pd.set_option('display.max_colwidth', None)

# Pre-set the path to the image  to save plots later
directory_to_img = os.path.join('..', '..', 'images')


# For reproducibility, we're going to fix set seed
seed = 2021


default_datetime = datetime(2023, 6, 1)
s = st.date_input('time start estimate', default_datetime, format="MM.DD.YYYY")
e = st.date_input('time end estimate', datetime.now(), format="MM.DD.YYYY")

tckr = st.text_input('input crypto coin name', 'BTC-USD')



start = s
end = e


# Choose the Bitcoin index
tckr = 'BTC-USD'

ticker = yf.Ticker(tckr)
df = ticker.history(start=start,
                    end=end,
                    interval="1d")

df = df.drop(['Dividends', 'Stock Splits'], axis=1)

# Change context to poster to increase font sizes
sns.set_context("talk", font_scale=1.3)

#Perform Dickey-Fuller test:
print ('Results of Augmented Dickey-Fuller Test:')
dftest = adfuller(df.Close, autolag='AIC')
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
print (dfoutput)

def test_stationarity(timeseries):
    #determine rolling statistics
    rolmean = pd.Series(timeseries).rolling(window=24).mean()#24 hours on each day
    rolstd = pd.Series(timeseries).rolling(window=24).std()
    print('result of dickey fuller test:')
    dftest = adfuller(timeseries,autolag = 'AIC')
    dfoutput = pd.Series(dftest[0:4],index = ['Test statistics', 'p-value', '#lags used', 'number of observation used'])
    for key,value in dftest[4].items():
        dfoutput['critical value (%s)'%key] = value
    print(dfoutput)

# Calculate price returns as daily percentage change using pct_change()
df['returns'] = 100 * df.Close.pct_change().dropna()

# Calculate log returns based on above formula
df['log_returns'] = np.log(df.Close/df.Close.shift(1))

df.isnull().sum()

df.dropna(inplace=True)

df['Sq_Returns']=df.returns.mul(df.returns)
df.head()

df.returns.describe()
df.log_returns.describe()

df['returns'].dropna(inplace=True)
test_stationarity(df['returns'])

# For log returns
adfuller_results = adfuller(df.log_returns.dropna())

print(f'ADF Statistic: {adfuller_results[0]}')
print(f'p-value: {adfuller_results[1]}')
print('Critical Values:')
for key, value in adfuller_results[4].items():
    print(f'{key}: {value:.4f}')

df.shape
data_return= df[["returns"]]

# Pre-determine desired test & validation sizes
test_size = 182


# Convert to indices
split_time_1 = len(data_return) - 182
split_time_2 = len(data_return) - 182

# Get corresponding datetime indices for each set
train_idx = df.index[:split_time_1]
test_idx = df.index[split_time_2:]

print(f'TRAINING \tFrom: {train_idx[0]} \tto: {train_idx[-1]} \t{len(train_idx)} days')
print(f'TEST \t\tFrom: {test_idx[0]} \tto: {test_idx[-1]} \t{len(test_idx)} days')

# Split returns into 2 parts (this would be the input for GARCH models)
r_train = data_return.returns[train_idx]
r_test = data_return.returns[test_idx]

r_train.describe()

r_train.describe()

print(data_return.shape)
train = data_return.iloc[:-182]
test = data_return.iloc[-182:]
print(train.shape,test.shape)

def hurst(ts):
    """Returns the Hurst Exponent of the time series vector ts"""
    # Create the range of lag values
    lags = range(2, 100)

    # Calculate the array of the variances of the lagged differences
    tau = [sqrt(std(subtract(ts[lag:], ts[:-lag]))) for lag in lags]

    # Use a linear fit to estimate the Hurst Exponent
    poly = polyfit(log(lags), log(tau), 1)

    # Return the Hurst exponent from the polyfit output
    return poly[0]*2.0

def plot_correlogram(x, lags=None, title=None):
    lags = min(10, int(len(x)/5)) if lags is None else lags
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    x.plot(ax=axes[0][0])
    q_p = np.max(q_stat(acf(x, nlags=lags), len(x))[1])
    stats = f'Q-Stat: {np.max(q_p):>8.2f}\nADF: {adfuller(x)[1]:>11.2f} \nHurst: {round(hurst(x.values),2)}'
    axes[0][0].text(x=0.1, y=1.5, s=stats, transform=axes[0][0].transAxes)
    probplot(x, plot=axes[0][1])
    mean, var, skew, kurtosis = moment(x, moment=[1, 2, 3, 4])
    s = f'Mean: {mean:>12.2f}\nSD: {np.sqrt(var):>16.2f}\nSkew: {skew:12.2f}\nKurtosis:{kurtosis:9.2f}'

    axes[0][1].text(x=.4, y=1.4, s=s, transform=axes[0][1].transAxes)

    plot_acf(x=x, lags=lags, zero=False, ax=axes[1][0])
    plot_pacf(x, lags=lags, zero=False, ax=axes[1][1])
    axes[1][0].set_xlabel('Lag')
    axes[1][1].set_xlabel('Lag')
    fig.suptitle(title, fontsize=20)
    fig.tight_layout()
    fig.subplots_adjust(top=0.9,hspace=1.5)

# Set seed for reproducibility
np.random.seed(seed)

model_garch_1 = arch_model(r_train, p=1, q=1)
result_1 = model_garch_1.fit(disp='off')
print()
print(result_1.summary())

model_garch_2 = arch_model(r_train, vol='GARCH', p=4, q=4)
result_2 = model_garch_2.fit(disp='off')
print()
print(result_2.summary())


model_garch_3 = arch_model(r_train, vol='GARCH', p=7, q=7)
result_3 = model_garch_3.fit(disp='off')
print()
print(result_3.summary())

def transform_volatility_to_scaler(scaler, tf_series):
    '''
    Transform a series to a fitted scaler
    '''
    idx = tf_series.index
    output = pd.Series(scaler.transform(tf_series.values.reshape(-1,1))[:,0],
                       index=idx)
    return output

def scale_tf_cond_vol(model_result):
    '''
    Scale & Transform Conditional Volatility
    Estimated by GARCH Models
    '''
    # Obtain estimated conditional volatility from model result
    cond_vol = model_result.conditional_volatility

    # Initiate scaler
    scaler = MinMaxScaler()

    # Fit scaler to model's estimated conditional volatility
    scaler = scaler.fit(cond_vol.values.reshape(-1,1))

    scaled_cond_vol = transform_volatility_to_scaler(scaler, cond_vol)
    return scaler, scaled_cond_vol


# Get volatility scaler & scaled conditional volatility from model result
scaler_garch, scaled_cond_vol = scale_tf_cond_vol(result_1)

n_future = 182
# viz_cond_vol(scaled_cond_vol, 'GARCH(1,1)')

# One step expanding window forecast
# Initializing rolling_forecast
rolling_forecasts = []
idx = df.index

# Iterate over each time step in the validation set
for i in range(len(test_idx)):
    # Get the data at all previous time steps
    idx = test_idx[i]
    train = df.returns[:idx]

    # Train model using all previous time steps' data
    model = arch_model(train, vol='GARCH', p=1, q=1,
                       dist='normal')
    model_fit = model.fit(disp='off')

    # Make prediction n_future days out
    vaR = model_fit.forecast(horizon=n_future,
                             reindex=False).variance.values
    # Get the sqrt of average n_future days variance
    pred = np.sqrt(np.mean(vaR))

    # Append to rolling_forecasts list
    rolling_forecasts.append(pred)

gm_1_preds = pd.Series(rolling_forecasts, index=test_idx)

# Transform predictions using fitted scaler
gm_1_preds_scaled = transform_volatility_to_scaler(scaler_garch, gm_1_preds)

# Define root mean squared percentage error function
def RMSPE(y_true, y_pred):
    """
    Compute Root Mean Squared Percentage Error between 2 arrays
    """
    output = np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))
    return output

# Define root mean squared error function
def RMSE(y_true, y_pred):
    """
    Compute Root Mean Squared Error between 2 arrays
    """
    output = np.sqrt(mse(y_true, y_pred))
    return output

# Initiate a dataframe for model performance tracking & comparison
perf_df = pd.DataFrame(columns=['Model', 'Validation RMSPE', 'Validation RMSE'])

# A function that logs model name, rmse and rmpse into perf_df for easy comparison later
def log_perf(y_true, y_pred, model_name):
    perf_df.loc[len(perf_df.index)] = [model_name,
                                       RMSPE(y_true, y_pred),
                                       RMSE(y_true, y_pred)]
    return perf_df

# Append metrics outputs to perf_df dataframe
log_perf(r_test, gm_1_preds_scaled,
         'GARCH(1,1), Constant Mean, Normal Dist')

# inspecting the residuals
gm_resid = result_1.resid
gm_std = result_1.conditional_volatility

# Standardizing residuals
gm_std_resid = gm_resid / gm_std

# Set seed for reproducibility
np.random.seed(seed)

gjr_gm = arch_model(r_train, p=1, q=1, o=1,
                    vol='GARCH', dist='skewt')
result_4 = gjr_gm.fit(disp='off')
print(result_4.summary())

# Get volatility scaler & scaled conditional volatility from model result
scaler_gjr, scaled_gjr_cond_vol = scale_tf_cond_vol(result_4)

# viz_cond_vol(scaled_gjr_cond_vol, 'GJR-GARCH(1,1)')

# Rolling window forecast
# Initializing rolling_forecasts values list
rolling_forecasts = []

# Iterate over each time step in the validation set
for i in range(len(test_idx)):
    # Get the data at all previous time steps
    idx = test_idx[i]
    train = df.returns[:idx].dropna()

    # Train model using all previous time steps' data
    model = arch_model(train, p=1, q=1, o=1,
                       vol='GARCH', dist='skewt')
    model_fit = model.fit(disp='off')

    # Make prediction n_future days out
    vaR = model_fit.forecast(horizon=n_future,
                             reindex=False).variance.values
    pred = np.sqrt(np.mean(vaR))

    # Append to rolling_forecasts list
    rolling_forecasts.append(pred)

gjr_1_preds = pd.DataFrame(rolling_forecasts, index=test_idx)

# Transform predictions using fitted scaler
gjr_1_preds_scaled = transform_volatility_to_scaler(scaler_gjr, gjr_1_preds)

# Append metrics outputs to perf_df dataframe
log_perf(r_test, gjr_1_preds_scaled,
         "Analytical GJR-GARCH(1,1,1), Constant Mean, Skewt Dist")


rolling_predictions = []
test_size = 182
for i in range(test_size):
  train = data_return[:-(test_size-i)]
  model = arch_model(train['returns'], p=1, q=1)
  model_fit = model.fit(disp='off')
  pred = model_fit.forecast(horizon=1)
  rolling_predictions.append(np.sqrt(pred.variance.values[-1,:][0]))


rolling_predictions = pd.Series(rolling_predictions, index=data_return["returns"].index[-182:])

# report performance
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
mse = mean_squared_error(test['returns'], rolling_predictions)
print('MSE: '+str(mse))
mae = mean_absolute_error(test['returns'], rolling_predictions)
print('MAE: '+str(mae))
rmse = math.sqrt(mean_squared_error(test['returns'], rolling_predictions))
print('RMSE: '+str(rmse))
mape = np.mean(np.abs(rolling_predictions - test['returns'])/np.abs(test['returns']))
print('MAPE: '+str(mape))

# Make 7-period ahead forecast
predd = result_1.forecast(horizon=7)
future_datess = [train["returns"].index[-1] + timedelta(days=i) for i in range(1,8)]
predd = pd.Series(np.sqrt(predd.variance.values[-1,:]), index=future_datess)

plt.figure(figsize=(12,4))
plt.plot(predd)
plt.title('Volatility Prediction - Next 7 Days', fontsize=20)
plt.show()

st.line_chart(predd)



