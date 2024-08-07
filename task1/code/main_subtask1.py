import os
import logging
import argparse
import xgboost as xgb
from typing import NoReturn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection as sk
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
import evaluation_scripts.eval_passengers_up as eval

TRAIN_BUS_CSV_PATH = "data/train_bus_schedule.csv"
X_PASSENGER = "data/X_passengers_up.csv"
X_TRIP = "data/X_trip_duration.csv"
ENCODER = "windows-1255"
RANDOM_STATE = 42
BASE_LINE_SAMPLE_SIZE = 0.05
MAX_DEPTH = 7
LEARNING_RATE = 0.1
N_ESTIMATORS = 100

X_COL = ['trip_id',
 'part',
 'trip_id_unique_station',
 'trip_id_unique',
 'line_id',
 'direction',
 'alternative',
 'cluster',
 'station_index',
 'station_id',
 'station_name',
 'arrival_time',
 'door_closing_time',
 'arrival_is_estimated',
 'latitude',
 'longitude',
 'passengers_continue',
 'mekadem_nipuach_luz',
 'passengers_continue_menupach']

WORDS_WEIGHT = ['אצ"ל',
                'ביאליק',
                'ההגנה',
                'חזון',
                'לוינסקי',
                'סוקולוב',
                'עקיבא',
                'קניון',
                'רבי',
                'רכבת',
                'אצ"ל',
                'ביאליק',
                'בלפור',
                'גשר',
                'ההגנה',
                'המלך',
                "ז'בוטינסקי",
                'חזון',
                'יוספטל',
                'כצנלסון',
                'לוינסקי',
                'סוקולוב',
                'עקיבא',
                'קניון',
                'רבי',
                'רוטשילד']


"""
usage:
    python code/main.py --training_set PATH --test_set PATH --out PATH

for example:
    python code/main.py --training_set /cs/usr/gililior/training.csv --test_set 
    /cs/usr/gililior/test.csv --out predictions/trip_duration_predictions.csv 

"""


# implement here your load,preprocess,train,predict,save functions (or any other design you choose)

def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = "feature_evaluation") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    # Ensure output path exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Ensure y is numeric
    y = pd.to_numeric(y, errors='coerce')

    for feature in X.columns:
        # Ensure feature column is numeric
        X[feature] = pd.to_numeric(X[feature], errors='coerce')
        
        # Drop rows with NaN values
        valid_data = X[[feature]].join(y).dropna()
        if valid_data.empty:
            logging.warning(f"No valid data for feature {feature}. Skipping plot.")
            continue

        # Calculate Pearson correlation
        pearson_corr = np.cov(valid_data[feature], valid_data[y.name])[0, 1] / (np.std(valid_data[feature]) * np.std(valid_data[y.name]))

        # Create scatter plot
        plt.figure(figsize=(10, 6))
        plt.scatter(valid_data[feature], valid_data[y.name], alpha=0.5)
        plt.title(f'{feature} vs people_on_bus\nPearson Correlation: {pearson_corr:.2f}', fontsize=14)
        plt.xlabel(feature, fontsize=12)
        plt.ylabel('Passengers', fontsize=12)
        plt.grid(True)

        # Save plot to file
        plot_filename = os.path.join(output_path, f'{feature}_vs_passengers.png')
        plt.savefig(plot_filename)
        plt.close()


def load_data(train_path, passenger_path,include_baseline = False):
    # train_encoding = detect_encoding(train_path)
    # test_encoding = detect_encoding(passenger_path)
    train_df = pd.read_csv(train_path, encoding=ENCODER)
    test_df = pd.read_csv(passenger_path, encoding=ENCODER)
    if include_baseline:
        sample_size = BASE_LINE_SAMPLE_SIZE # Adjust as needed
        baseline = train_df.sample(frac=sample_size, random_state=RANDOM_STATE)
        remaining_data = train_df.drop(baseline.index)
        remaining_data.to_csv("remaining_data.csv")
    return train_df,test_df


def preprocessing_baseline(X: pd.DataFrame, y: pd.Series):
    # Save the trip_id_unique_station column
    trip_id_unique_station = X["trip_id_unique_station"].copy()

    # Convert to datetime
    X = X.copy()  # To avoid SettingWithCopyWarning
    X['door_closing_time'] = pd.to_datetime(X['door_closing_time'], errors='coerce')
    X['arrival_time'] = pd.to_datetime(X['arrival_time'], errors='coerce')

    # Verify the columns are datetime
    if not pd.api.types.is_datetime64_any_dtype(X['door_closing_time']):
        raise TypeError("door_closing_time column is not datetime type")
    if not pd.api.types.is_datetime64_any_dtype(X['arrival_time']):
        raise TypeError("arrival_time column is not datetime type")

    # Create door delta columns
    X["door_close_delta"] = None
    mask_notna = X["door_closing_time"].notna()
    X.loc[mask_notna, 'door_close_delta'] = (X.loc[mask_notna, 'door_closing_time'] - X.loc[mask_notna, 'arrival_time']).dt.total_seconds()

    # Handle NaN values by replacing them with the mean
    door_delta_mean = X["door_close_delta"].mean()
    X["door_close_delta"] = X["door_close_delta"].fillna(door_delta_mean)

    # Categorize arrival time
    arrival_hours = X['arrival_time'].dt.hour
    percentiles = arrival_hours.describe(percentiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    percentile_values = percentiles.loc[
        ['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%']
    ].values
    labels = [f'{int(value)}' for value in percentile_values]
    labels.insert(0, '0')
    X['arrival_time_label'] = pd.cut(arrival_hours,
                                     bins=[0] + list(percentile_values) + [24],
                                     labels=labels,
                                     include_lowest=True)

    # Label Encoding
    label_encoder = LabelEncoder()
    X['part_encoded'] = label_encoder.fit_transform(X['part'])
    X['alternative_encoded'] = label_encoder.fit_transform(X['alternative'])

    # Drop unnecessary columns
    X = X.drop(columns=["arrival_time", "door_closing_time", "cluster", "station_name",
                        "part", "trip_id_unique", "alternative"])

    # Ensure all remaining columns are numeric
    for column in X.columns:
        if X[column].dtype == 'object':
            X[column] = X[column].apply(pd.to_numeric, errors='coerce')

    # Split the data
    X_train, X_test, y_train, y_test = sk.train_test_split(X, y, test_size=0.25,
                                                           random_state=RANDOM_STATE)

    # Add the trip_id_unique_station column back to X_train and X_test
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train["trip_id_unique_station"] = trip_id_unique_station.loc[X_train.index]
    X_test["trip_id_unique_station"] = trip_id_unique_station.loc[X_test.index]

    return X_train, X_test, y_train, y_test


def desition_trees(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series,
                   y_test: pd.Series):
    # Ensure X_test has trip_id_unique_station column
    if 'trip_id_unique_station' not in X_test.columns:
        raise ValueError("X_test must contain 'trip_id_unique_station' column")

    # Drop the 'trip_id_unique_station' column from X_train and X_test
    X_train = X_train.drop(columns=['trip_id_unique_station'])
    X_test = X_test.drop(columns=['trip_id_unique_station'])

    # Convert all columns to numeric if necessary
    for col in X_train.columns:
        X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
        X_test[col] = pd.to_numeric(X_test[col], errors='coerce')

    # Linear regression
    model = DecisionTreeRegressor(random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Create DataFrames for evaluation
    predictions_df = pd.DataFrame({
        'trip_id_unique_station': X_test.index,
        # Ensure we have the correct index for trip_id_unique_station
        'passengers_up_x': y_pred
    })

    ground_truth_df = pd.DataFrame({
        'trip_id_unique_station': X_test.index,
        # Ensure we have the correct index for trip_id_unique_station
        'passengers_up_y': y_test.values
    })

    # Evaluate using eval_boardings
    mse_board = eval.eval_boardings(predictions_df, ground_truth_df)
    return mse_board


def linear_regression(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series,
                      y_test: pd.Series):
    # Ensure X_test has trip_id_unique_station column
    if 'trip_id_unique_station' not in X_test.columns:
        raise ValueError("X_test must contain 'trip_id_unique_station' column")

    # Drop the 'trip_id_unique_station' column from X_train and X_test
    X_train = X_train.drop(columns=['trip_id_unique_station'])
    X_test = X_test.drop(columns=['trip_id_unique_station'])

    # Convert all columns to numeric if necessary
    for col in X_train.columns:
        X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
        X_test[col] = pd.to_numeric(X_test[col], errors='coerce')

    # Linear regression
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Create DataFrames for evaluation
    predictions_df = pd.DataFrame({
        'trip_id_unique_station': X_test.index,  # Ensure we have the correct index for trip_id_unique_station
        'passengers_up_x': y_pred
    })

    ground_truth_df = pd.DataFrame({
        'trip_id_unique_station': X_test.index,  # Ensure we have the correct index for trip_id_unique_station
        'passengers_up_y': y_test.values
    })

    # Evaluate using eval_boardings
    mse_board = eval.eval_boardings(predictions_df, ground_truth_df)
    return mse_board


def polynomial_fitting(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series,
                       y_test: pd.Series):
    # Ensure X_test has trip_id_unique_station column
    if 'trip_id_unique_station' not in X_test.columns:
        raise ValueError("X_test must contain 'trip_id_unique_station' column")

    # Polynomial fitting
    poly = PolynomialFeatures(degree=2)
    X_train_poly = poly.fit_transform(X_train.drop(columns=['trip_id_unique_station']))
    X_test_poly = poly.transform(X_test.drop(columns=['trip_id_unique_station']))

    # Initialize and train the Polynomial Regression model
    model_poly = LinearRegression()
    model_poly.fit(X_train_poly, y_train)

    # Predict on the test set
    y_pred_poly = model_poly.predict(X_test_poly)

    # Create DataFrames for evaluation
    predictions_df = pd.DataFrame({
        'trip_id_unique_station': X_test['trip_id_unique_station'],
        'passengers_up_x': y_pred_poly
    })

    ground_truth_df = pd.DataFrame({
        'trip_id_unique_station': X_test['trip_id_unique_station'],
        'passengers_up_y': y_test.values
    })

    # Evaluate using eval_boardings
    mse_board = eval.eval_boardings(predictions_df, ground_truth_df)
    csv_output(mse_board, y_pred_poly, X_test['trip_id_unique_station'])
    return mse_board

def csv_output(passengers_up: pd.Series, trip_id_unique_station, path):
    # Create DataFrame with predictions
    predictions_df = pd.DataFrame({
        'trip_id_unique_station': trip_id_unique_station,
        'passengers_up': passengers_up.round()
    })

    # Save predictions to CSV file
    predictions_df.to_csv(path, index=False)


def xg_boost(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series):
    # for params in parameters:
        # Create XGBoost model
    params = {'max_depth': MAX_DEPTH, 'learning_rate': LEARNING_RATE, 'n_estimators': N_ESTIMATORS}
    model = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse', **params)
    # Train the model
    model.fit(X_train.drop(columns=['trip_id_unique_station']), y_train)
    # Predict on test set
    y_pred = model.predict(X_test.drop(columns=['trip_id_unique_station']))
    # Calculate RMSE
    # rmse = mean_squared_error(y_test, y_pred, squared=False)
    # results.append((params, rmse))

    return y_pred



def preprocessing_data(X, is_train):
    if is_train:
        xgboost_X =X[X_COL+["passengers_up"]]
    else:
        xgboost_X =X[X_COL]

    for i, word in enumerate(WORDS_WEIGHT, start=1):
        xgboost_X[f'x_{i}'] = xgboost_X['station_name'].str.contains(word, regex=False).astype(int)

    label_encoder = LabelEncoder()
    xgboost_X['cluster'] = label_encoder.fit_transform(xgboost_X['cluster'])
    xgboost_X['alternative'] = label_encoder.fit_transform(xgboost_X['alternative'])
    xgboost_X['part'] = label_encoder.fit_transform(xgboost_X['part'])
    xgboost_X['station_id'] = label_encoder.fit_transform(xgboost_X['station_id'])
    xgboost_X['trip_id_unique'] = label_encoder.fit_transform(xgboost_X['trip_id_unique'])
    xgboost_X['arrival_is_estimated'] = xgboost_X['arrival_is_estimated'].astype(int)
    xgboost_X['door_closing_time'] = pd.to_datetime(xgboost_X['door_closing_time'])
    xgboost_X['arrival_time'] = pd.to_datetime(xgboost_X['arrival_time'])

    # Create door delta columns
    xgboost_X["door_close_delta"] = None
    mask_notna = xgboost_X["door_closing_time"].notna()
    xgboost_X.loc[mask_notna, 'door_close_delta'] = (
                xgboost_X.loc[mask_notna, 'door_closing_time'] - xgboost_X.loc[
            mask_notna, 'arrival_time']).dt.total_seconds()
    door_delta_mean = xgboost_X["door_close_delta"].mean()
    xgboost_X["door_close_delta"] = xgboost_X["door_close_delta"].fillna(door_delta_mean)

    # Categorize arrival time
    arrival_hours = xgboost_X['arrival_time'].dt.hour
    percentiles = arrival_hours.describe(percentiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    percentile_values = percentiles.loc[
        ['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%']
    ].values
    labels = [f'{int(value)}' for value in percentile_values]
    labels.insert(0, '0')
    xgboost_X['arrival_time_label'] = pd.cut(arrival_hours,
                                             bins=[0] + list(percentile_values) + [24],
                                             labels=labels,
                                             include_lowest=True)

    if is_train:
        index_to_drop = xgboost_X["passengers_up"][xgboost_X["passengers_up"] > 30].index
        xgboost_X = xgboost_X.drop(index_to_drop)

    xgboost_X["arrival_time_label"] = xgboost_X["arrival_time_label"].astype(int)
    del xgboost_X["arrival_time"]
    del xgboost_X["door_closing_time"]
    del xgboost_X["station_name"]
    # del xgboost_X["trip_id_unique_station"]
    return xgboost_X 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="XGBoost model training and evaluation")
    parser.add_argument('--training_set', type=str, required=True,
                        help="Path to the training set CSV file")
    parser.add_argument('--test_set', type=str, required=True,
                        help="Path to the passenger data CSV file")
    parser.add_argument('--out', type=str, required=True,
                        help="Path to save the output (predictions or results)")
    args = parser.parse_args()
    train, test = load_data(args.training_set, args.test_set)
    train_processed = preprocessing_data(train,is_train = True)
    X_train , y_train = train_processed.drop(columns = "passengers_up") , train_processed["passengers_up"]
    X_test = preprocessing_data(test,is_train = False)
    result = xg_boost(X_train, X_test, y_train)
    csv_output(result, X_test["trip_id_unique_station"], args.out)

