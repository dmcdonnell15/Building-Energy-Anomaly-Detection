import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score
)
from xgboost import XGBClassifier
# read data
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from prophet import Prophet
import seaborn as sns
from joblib import Parallel, delayed


def plot_anomalies(df, building_id=None, anomaly_label_actual=None, anomaly_label_predicted=None, pred_meter_reading=None, start_time=None, end_time=None):
    """
    Take any building you want and plot what it looks like with the predicted anomalies and/or predicted meter_readings labeled
    """
    # filter for one building
    if building_id is None:
        building_id = df['building_id'].sample(1).iloc[0] # random building ID
        
    building_data = df[df['building_id'] == building_id].copy()

    # filter for specific timeframe
    if start_time is not None:
        building_data = building_data[building_data['timestamp'] >= start_time]
    if end_time is not None:
        building_data = building_data[building_data['timestamp'] <= end_time]

    if pred_meter_reading is not None:
        ax = building_data.plot(x='timestamp', y=['meter_reading', pred_meter_reading], figsize=(20,6), title=f'meter readings for building {building_id}')
    else:
        ax = building_data.plot(x='timestamp', y='meter_reading', figsize=(20,6), title=f'meter readings for building {building_id}')

    # Plot actual anomalies
    if anomaly_label_actual:
        anomaly_data_act = building_data.copy()
        anomaly_data_act.loc[anomaly_data_act[anomaly_label_actual] != 1, 'meter_reading'] = np.nan
    
        # Plot only the anomaly points
        if not anomaly_data_act['meter_reading'].isna().all(): # Check if there are any non-NaN values
            anomaly_data_act.plot(x='timestamp', y='meter_reading', kind='scatter', 
                            color='red', marker = 'o', ax=ax, label=f'{anomaly_label_actual}', s=30)
    
    # Plot predicted anomalies
    if anomaly_label_predicted is not None:
        anomaly_data_pred = building_data.copy()
        anomaly_data_pred.loc[anomaly_data_pred[anomaly_label_predicted] != 1, 'meter_reading'] = np.nan
        
        # Plot only the anomaly points
        if not anomaly_data_pred['meter_reading'].isna().all(): # Check if there are any non-NaN values
            anomaly_data_pred.plot(x='timestamp', y='meter_reading', kind='scatter', 
                            color='green', marker = '*', ax=ax, label=f'{anomaly_label_predicted}', s=50)
        
    return ax

def eval_metrics(df, pred_anomaly, building_id=None):
    """
    Take a dataframe and calculate the accuracy, precision, recall, F1 score, and AUC
    """
    eval_df = df[df['building_id']==building_id].copy() if building_id is not None else df.copy()

    accuracy = accuracy_score(eval_df['anomaly'], eval_df[pred_anomaly])
    precision = precision_score(eval_df['anomaly'], eval_df[pred_anomaly])
    recall = recall_score(eval_df['anomaly'], eval_df[pred_anomaly])
    f1 = f1_score(eval_df['anomaly'], eval_df[pred_anomaly])
    
    if building_id is not None:
        print('Building ID: ', building_id)
    print('Accuracy: ', round(accuracy * 100, 2), '%')
    print('Recall: ', round(recall * 100, 2), '%')
    print('Precision: ', round(precision * 100, 2), '%')
    print('F1 Score: ', round(f1 * 100, 2), '%')

def evaluate_ts_predictions(actual, predicted, excluded_vals=[1.0]):
    """
    For time series model, determine the MAPE
    """
    mask = ~np.isnan(actual) # exclude NaN
    for val in excluded_vals:
        mask = mask & (actual != val) & (actual > 1.0)

    actual_valid = actual[mask]
    predicted_valid = predicted[mask]

    mape = np.mean(abs((actual_valid - predicted_valid) / actual_valid)) * 100

    print('MAPE: ', mape)

def create_submission_df(df, model_name, cols_to_use, row_id_label='row_id', anomaly_score_label='pred_anomaly_score', anomaly_label='pred_anomaly'):
    """
    Takes test dataframe with anomaly score predictions and prepares columns needed to submit
    """
    df_copy = df.copy()
    df_for_predictions = df_copy[cols_to_use].copy()
    proba = model_name.predict_proba(df_for_predictions)[:, 1]
    preds = model_name.predict(df_for_predictions)
    df_copy[anomaly_score_label] = proba
    df_copy[anomaly_label] = preds
    submission_df = df_copy[[row_id_label, anomaly_score_label]].rename(columns={row_id_label: 'row_id', anomaly_score_label: 'anomaly'}).copy()
    return df_copy, submission_df

def label_anomalies_iqr(df):
    """
    Takes a dataset of buildings and meter readings. For each building, calculates Q3/Q1/IQR and upper/lower limits 
    of where outliers are. It then uses those limits to label a new column called pred_anomaly_iqr as 1/0.
    """
    # do the calcs
    readings = df['meter_reading']
    q3 = readings.quantile(0.75)
    q1 = readings.quantile(0.25)

    IQR = q3 - q1
    factor = 1.5

    upper_limit = q3 + (IQR * factor)
    lower_limit = q1 - (IQR * factor)

    # create new column label
    df['pred_anomaly_iqr'] = np.where((df['meter_reading'] > upper_limit)
                                                | (df['meter_reading'] < lower_limit)
                                                , 1, 0)
        
    return df

def anomalies_iqr(df):
    """
    Create a pred_anomaly_iqr label using the IQR to determine outliers
    """
    df_with_anomalies = df.groupby('building_id').apply(label_anomalies_iqr, include_groups=False)
    df_with_anomalies.reset_index(inplace=True, level=0)
    return df_with_anomalies

def iforest_model(df):
    """
    Takes the train dataframe and builds an iforest model for each building
    """
    # create features for month/day/hour
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month

    # instantiate an iforest model and scaler for the features
    iforest_df = pd.DataFrame()
    scaler = StandardScaler()
    iso_forest = IsolationForest(n_estimators=150, contamination=0.025, random_state=1)
    features_to_scale = ['hour', 'day', 'month', 'meter_reading_imputed']

    # loop through each building
    for bldg in df['building_id'].unique():
        bldg_df = df[df['building_id']==bldg].copy()
        # impute missing meter readings with mean values
        bldg_df['meter_reading_imputed'] = bldg_df['meter_reading'].fillna(bldg_df['meter_reading'].mean())
        # scale the features using standardization
        X = scaler.fit_transform(bldg_df[features_to_scale])
        # predict anomaly scores / anomalies with iso forest
        iso_forest.fit(X)
        bldg_df['pred_anomaly_iforest'] = (iso_forest.predict(X) == -1).astype(int) # iso forest labels anomalies as -1
        bldg_df['anomaly_score'] = -iso_forest.score_samples(X) # in iso forest more negative scores = more likely to be an anomaly 
        # concat dataframes together into final df
        iforest_df = pd.concat([iforest_df, bldg_df])
    return iforest_df

def feature_preparation(df, impute_meter_readings=True, impute_other_cols=True, additional_drop_cols=[]):
    """
    impute columns / add 1/0 categorical columns for missing data if columns are used
    drop columns not needed
    optional param to drop additional columns
    """
    if impute_meter_readings:
        # create 1/0 col for missing meter readings and impute meter readings with building mean
        df['missing_meter_reading'] = df['meter_reading'].isna().astype(int)
        building_means = df.groupby(['building_id'])['meter_reading'].transform('mean')
        df['meter_reading_imputed'] = df['meter_reading'].fillna(building_means)

    if impute_other_cols:
        # Cols year_built, cloud_coverage likely have null values represented by 255, and wind_direction likely has null value at 65535. Make 1/0 cols and imputed cols for each
        null_indicators = {
            'year_built': 255,
            'cloud_coverage': 255,
            'wind_direction': 65535
        }

        for col, null_val in null_indicators.items():
            # create 1/0 for missing vals
            df[f'missing_{col}'] = (df[col]==null_val).astype(int)

            # impute cols with mean value for year_built, cloud_coverage, and wind_direction
            mean = df[df[col] != null_val][col].mean()
            df[f'{col}_imputed'] = df[col].replace(null_val, mean)

    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category')

    # cols that are already taken into consideration by other columns
    cols_not_needed = ['year', 'gte_meter', 'building_weekday_hour', 'building_weekday', 'building_month', 'building_hour', 'building_meter', 'year_built', 'cloud_coverage', 'wind_direction']

    # cols that have high correlation
    cols_hicorr = ['meter_reading', 'timestamp', 'gte_meter_hour', 'gte_meter_weekday', 'gte_meter_month', 'gte_meter_building_id', 'gte_meter_primary_use', 'gte_meter_site_id',
                    'gte_meter_building_id_hour', 'gte_meter_building_id_weekday', 'gte_meter_building_id_month', 'air_temperature_max_lag7',
                    'air_temperature_min_lag7', 'air_temperature_std_lag7', 'air_temperature_max_lag73','air_temperature_min_lag73', 'air_temperature_std_lag73']
                    # , 'primary_use', 'weekday_hour'] #temporary drop because they are objects causing problems
    
    # Drop columns not used. Includes cols that are just concatenated data, cols with high correlation, and optional additional cols that are not used.
    df = df.drop(cols_not_needed + cols_hicorr + additional_drop_cols, axis=1)

    return df

def xgb_model_custom(df, df_full, cols_to_use=[], target=['anomaly'], n_estimators=100, max_depth=5, learning_rate=0.1, smote=False, plot_learning_curve=True):
    """
    Train and fit an XGB model, and return evaluation metrics
    """
    # make sure train/test dataset doesn't have overlapping buildings
    buildings = df['building_id'].unique()

    train_buildings, test_buildings = train_test_split(buildings, test_size=0.2, random_state=15)

    # train cols are all cols minus and building_id anomalies
    if cols_to_use == []:
        cols_to_use = df.drop(columns=['building_id', 'anomaly']).columns.to_list()

    X_train = df[df['building_id'].isin(train_buildings)][cols_to_use]
    X_test = df[df['building_id'].isin(test_buildings)][cols_to_use]
    y_train = df[df['building_id'].isin(train_buildings)][target]
    y_test = df[df['building_id'].isin(test_buildings)][target]
    
    # optional SMOTE
    if smote==True:
        smote = SMOTE(sampling_strategy=0.5, random_state=15)
        X_train, y_train = smote.fit_resample(X_train, y_train)

    # create evaluation set for learning curves
    eval_set = [(X_train, y_train), (X_test, y_test)]

    # fit the model
    xgb_full = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, enable_categorical=True, objective='binary:logistic', eval_metric='auc')
    xgb_full.fit(X_train, y_train,
                 eval_set=eval_set,
                 verbose=False)
    
    # After fitting the model, get feature importances with their names
    feature_importance_df = pd.DataFrame({
        'feature': cols_to_use,  # your feature names
        'importance': xgb_full.feature_importances_
    })

    # make predictions on holdout set and append to holdout dataframe
    preds = xgb_full.predict(X_test)
    proba = xgb_full.predict_proba(X_test)[:, 1]
    X_test['pred_anomaly_xgb'] = preds
    X_test['anomaly_score'] = proba
    X_test_with_anom = pd.concat([X_test, y_test], axis=1)
    
    # make predictions on full training dataset and append to original df
    df_train_with_predictions = df_full.copy()
    preds_full = xgb_full.predict(df_train_with_predictions[cols_to_use])
    proba_full = xgb_full.predict_proba(df_train_with_predictions[cols_to_use])[:, 1]
    df_train_with_predictions['pred_anomaly_xgb'] = preds_full
    df_train_with_predictions['anomaly_score'] = proba_full

    # Plot learning curves
    if plot_learning_curve:
        results = xgb_full.evals_result()
        
        plt.figure(figsize=(10, 6))
        plt.plot(results['validation_0']['auc'], label='Train AUC')
        plt.plot(results['validation_1']['auc'], label='Test AUC')
        plt.xlabel('Number of Trees')
        plt.ylabel('AUC Score')
        plt.title('XGBoost Learning Curves')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # Print best iteration
        print(f"Best test AUC: {max(results['validation_1']['auc']):.4f}")

    # print evaluation metrics on test set
    auc = roc_auc_score(X_test_with_anom['anomaly'], X_test_with_anom['anomaly_score'])
    print(f'Model Params: n_estimators={n_estimators}, max_depth={max_depth}, learning_rate={learning_rate}, smote={smote}')
    print('AUC: ', auc)
    eval_metrics(X_test_with_anom, pred_anomaly='pred_anomaly_xgb')

    return xgb_full, df_train_with_predictions, feature_importance_df

def add_zscore_col(df):
    """
    Take dataframe grouped by building, output dataframe with additional residual and residual_zscore column which is how many std the zscore is away from the rest of the bldg zscores
    """
    try:

        df_prophet = df.copy()

        # rename columns to make it prophet friendly
        df_prophet = df_prophet.rename(columns={'timestamp': 'ds', 'meter_reading': 'y'})
        
        # initialize the prophet model
        m = Prophet(
            weekly_seasonality=15,
            daily_seasonality=15,  # increase number of Fourier terms for daily seasonality
            changepoint_prior_scale=0.5  # make the trend more flexible
            )
        
        good_data = df_prophet[
            (df_prophet['y'].notna()) &
            (df_prophet['y']!=1.0)
        ]

        m.fit(good_data)
        forecast = m.predict(df_prophet[['ds']])

        # drop forecasts below 0
        forecast['yhat'] = np.where(forecast['yhat'] < 0, 0, forecast['yhat'])

        # original df add yhat
        df = df.reset_index(drop=True)
        df['yhat'] = forecast['yhat'].reset_index(drop=True)

        # calculate residual and residual z_score
        df['residual'] = abs(df['meter_reading'] - df['yhat'])
        df['residual_zscore'] = (df['residual'] - df['residual'].mean()) /  df['residual'].std()

        return df
    
    except Exception as e:
        print(f"Error processing building {df['building_id'].iloc[0]}: {str(e)}")
        return None

def parallel_process_buildings(df, n_jobs=8):
    """
    Trains prophet model to predict meter readings for each individual building in parallel
    """
    # create list of grouped dfs for each building_id
    building_groups = [group for _, group in df.groupby('building_id')]
    
    # 8 parallel processes (one for each core on my computer) to run add_zscore_col function in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(add_zscore_col)(group) for group in building_groups
    )
    
    # Filter out None results and combine
    results = [r for r in results if r is not None]

    # returns all results combined into one df, or an empty dataframe if no results
    return pd.concat(results) if results else pd.DataFrame()


