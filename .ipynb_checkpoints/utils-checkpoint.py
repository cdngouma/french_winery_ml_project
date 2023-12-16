import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from pydlm import dlm, dynamic, trend

from statsmodels.tsa.stattools import adfuller, kpss


# Make data stationary
def make_stationary(df, column_name, lag=1):
    df_stationary = df.copy()
    df_stationary[column_name] = df[column_name].diff(lag)
    df_stationary = df_stationary.iloc[lag:]
    return df_stationary[column_name]


def back_transform(original_value, differenced_data, lag=1):
    back_transformed_column = pd.Series(index=range(len(differenced_data)))
    back_transformed_column.iloc[0] = original_value
    
    for i in range(lag, len(differenced_data)):
        v = differenced_data[i-1] + back_transformed_column.iloc[i-1]
        back_transformed_column.iloc[i] = v
    
    return back_transformed_column.values


# Augmented Dickey-Fuller (ADF) Test to check if data is stationary
def adf_test(timeseries, threshold=0.05, verbose=0):
    result = adfuller(timeseries, autolag='AIC')
    if verbose >= 1:
        print('ADF Statistic: {:.3f}'.format(result[0]))
        print('p-value: {:.3f}'.format(result[1]))
    if verbose >= 2:
        print('Critical Values:')
        for key, value in result[4].items():
            print('\t{}: {:.3f}'.format(key, value))
    if verbose >= 1:
        if result[1] <= threshold:
            print("\nADF test: Data is stationary")
        else:
            print("\nADF test: Data is non-stationary")
    return result[0], result[1]


# Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test to check if data is stationary
def kpss_test(timeseries, threshold=0.05, verbose=0):
    result = kpss(timeseries)
    if verbose >= 1:
        print('\nKPSS Statistic: {:.3f}'.format(result[0]))
        print('p-value: {:.3f}'.format(result[1]))
    if verbose >= 2:
        print('Critical Values:')
        for key, value in result[3].items():
            print('\t{}: {:.3f}'.format(key, value))
    if verbose >= 1:
        if result[1] <= threshold:
            print("\nKPSS test: Data is non-stationary")
        else:
            print("\nKPSS test: Data is stationary")
    return result[0], result[1]


def calculate_correlation(dataframe):
    # Calculate correlation matrix for the given DataFrame
    correlation_matrix = dataframe.corr()
    return correlation_matrix


def visualize_correlation(correlation_matrix, figsize=(8,6)):
    # Visualize the correlation matrix using a heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(correlation_matrix, annot=True, cmap='Blues', fmt='.2f', linewidths=.5)
    plt.title('Correlation Between Variables')
    #plt.xlabel('Time Series')
    #plt.ylabel('Time Series')
    plt.show()


# Split dataset into train and test sets
def split_dataset(data, test_size=0.2):
    data_sorted = data.sort_index()
    
    # Split the dataset
    train_data, test_data = train_test_split(data_sorted, test_size=test_size, shuffle=False)
    
    return train_data, test_data


# Fill missing values
def linear_interpolation(df, columns, window=None):
    df_ = df.copy()
    for col in columns:
        values_ = df_[col].values
        # Find the indices of known values
        known_indices = np.where(~np.isnan(values_))[0]
    
        # Interpolate between the first and last known values
        first_known_index = known_indices[0]
        if window:
            last_known_index = known_indices[min(first_known_index+window, len(known_indices)-1)]
        else:
            last_known_index = known_indices[-1]
    
        # Number of missing values to interpolate
        num_missing = last_known_index - first_known_index
    
        # Calculate the step between values
        step = (values_[last_known_index] - values_[first_known_index]) / (num_missing + 1)
    
        # Perform linear interpolation
        interpolated_values = []
        for i in range(0, first_known_index):
            interpolated_value = values_[first_known_index] - step * (first_known_index - i)
            interpolated_values.append(interpolated_value)
        
        values_[:first_known_index] = interpolated_values
        
        df_[col] = values_
    
    return df_


# Calculate MAE, MSE, and R-squared
def calculate_metrics(predicted, true):
    # Calculate Mean Absolute Error (MAE)
    mae = mean_absolute_error(true, predicted)

    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(true, predicted)
    
    # Calculate R-squared
    r_squared = r2_score(true, predicted)
    
    return mae, mse, r_squared


# Make columns stationary
def auto_make_stationary(data, column):
    for lag in range(1,101):
        stationary_column = make_stationary(df=data, column_name=column, lag=lag)
        _, p_value = adf_test(stationary_column, verbose=0)
        if p_value <= 0.05:
            print(f"'column' made stationary with lag={lag}")
            return stationary_column, lag
        if lag == 100:
            print(f"Warning: Failed to make '{column}' data stationary")
            stationary_column = data.iloc[lag:][column]
    return stationary_column, 0


# Perform model cross validation
def stats_models_cv(model_name, data, params={}, stationary_column=None, n_folds=5):
    tscv = TimeSeriesSplit(n_splits=n_folds)
    # Initialize list of metrics
    mae_scores = []
    mse_scores = []
    r2_scores = []
    
    original_mae_scores = []
    original_mse_scores = []
    original_r2_scores = []
    
    min_train_size = len(data)
    max_train_size = 0
    min_test_size = len(data)
    max_test_size = 0
    
    data_ = data.copy()
    original_data = None
    
    if stationary_column:
        stationarized_column, stationary_lag = auto_make_stationary(data_, stationary_column)
        data_[stationary_column] = stationarized_column
        data_ = data_.iloc[stationary_lag:]
        # Preserve original data
        original_data = data.copy()
    
    for train_idx, test_idx in tscv.split(data_):
        # Split data into train and test sets
        train_data, test_data = data_.iloc[train_idx].copy(), data_.iloc[test_idx].copy()
        if stationary_column:
            original_train, original_test = original_data.iloc[train_idx].copy(), original_data.iloc[test_idx].copy()
        
        # Record train and test sizes for stats
        min_train_size = min(min_train_size, len(train_data))
        max_train_size = max(max_train_size, len(train_data))
        min_test_size = min(min_test_size, len(test_data))
        max_test_size = max(max_test_size, len(test_data))
        
        # Initialize model
        if model_name == "holt-winters":
            # Select a value for seasonal periods
            default = max(2, len(train_data)//2)
            seasonal_periods = params.get("seasonal_periods", default)
            if seasonal_periods is None or seasonal_periods == "half":
                seasonal_periods = default
            elif seasonal_periods == "third":
                seasonal_periods = max(2, len(train_data)//3)
            
            # Initialize model
            model = ExponentialSmoothing(train_data, trend='add', seasonal='add', seasonal_periods=seasonal_periods)
            # Fit model on training data
            model_fit = model.fit()
            # Forecast next steps
            forecast = model_fit.forecast(steps=len(test_data))
            actual = test_data
            if stationary_column:
                original_forecast = back_transform(original_train.iloc[-1], forecast)
                original_actual = original_test
        elif model_name == "arima":
            p, d, q = params.get("p", 1), params.get("d", 1), params.get("q", 1)
            model = ARIMA(train_data, order=(p, d, q))
            # Fit model on training data
            model_fit = model.fit()
            # Forecast next steps
            forecast = model_fit.forecast(steps=len(test_data))
            actual = test_data
            if stationary_column:
                original_forecast = back_transform(original_train.iloc[-1], forecast)
                original_actual = original_test
        elif model_name == "dlm":
            discount = params.get("discount", 0.5)
            features_data = params.get("features_data", "linear")
            if features_data == "linear":
                components = trend(degree=1, discount=discount)
            else:
                components = dynamic(features=features_data, discount=discount)
            # Fit model on train data
            dlm_model = dlm(train_data.values)
            dlm_model = dlm_model + components
            dlm_model.fit()
            # Forecast the next steps
            forecast = dlm_model.predictN(N=len(test_data))[0]
            actual = test_data
            if stationary_column:
                original_forecast = back_transform(original_train.iloc[-1], forecast)
                original_actual = original_test
        elif model_name == "lr":
            lr_model = Ridge()
            lr_model.fit(train_data.values[:,:-1], train_data.values[:,-1])
            # Predictions
            forecast = lr_model.predict(test_data.values[:,:-1])
            actual = test_data.values[:,-1]
            if stationary_column:
                original_forecast = back_transform(original_train.values[-1, :-1], forecast)
                original_actual = original_test.values[:,-1] if original_test else None
        else:
            raise Exception(f"Model '{model_name}' is not supported")
        
        # Calculate metrics
        mae = mean_absolute_error(actual, forecast)
        mse = mean_squared_error(actual, forecast)
        r2 = r2_score(actual, forecast)
    
        # Append scores to lists
        mae_scores.append(mae)
        mse_scores.append(mse)
        r2_scores.append(r2)
        
        if stationary_column:
            original_mae = mean_absolute_error(original_actual, original_forecast)
            original_mse = mean_squared_error(original_actual, original_forecast)
            original_r2 = r2_score(original_actual, original_forecast)
        
            original_mae_scores.append(original_mae)
            original_mse_scores.append(original_mse)
            original_r2_scores.append(original_r2)
    
    # Print average metrics across all folds
    print(f"Mean MAE: {np.mean(mae_scores)}")
    print(f"Mean MSE: {np.mean(mse_scores)}")
    print(f"Mean R-squared: {np.mean(r2_scores)}")
    print(f"Train sizes: min={min_train_size}, max={max_train_size}")
    print(f"Test sizes: min={min_test_size}, max={max_test_size}")
    
    print("\nMetrics in non-stationary context")
    if stationary_column:
        # Print average metrics across all folds
        print(f"Mean MAE: {np.mean(original_mae_scores)}")
        print(f"Mean MSE: {np.mean(original_mse_scores)}")
        print(f"Mean R-squared: {np.mean(original_r2_scores)}")