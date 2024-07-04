import os
from argparse import ArgumentParser
import logging
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

"""
usage:
    python code/main.py --training_set PATH --test_set PATH --out PATH

for example:
    python code/main.py --training_set /cs/usr/gililior/training.csv --test_set 
    /cs/usr/gililior/test.csv --out predictions/trip_duration_predictions.csv 

"""


# implement here your load,preprocess,train,predict,save functions (or any other design you choose)

def feature_evaluation(X: pd.DataFrame, y: pd.Series,
                       output_path: str = "feature_evaluation") -> NoReturn:
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

    for feature in X.columns:
        # calculate Pearson correlation
        pearson_corr = np.cov(X[feature], y)[0, 1] / (np.std(X[feature]) * np.std(y))

        # Create scatter plot
        plt.figure(figsize=(10, 6))
        plt.scatter(X[feature], y, alpha=0.5)
        plt.title(f'{feature} vs people_on_bus\nPearson Correlation: {pearson_corr:.2f}',
                  fontsize=14)
        plt.xlabel(feature, fontsize=12)
        plt.ylabel('Price', fontsize=12)
        plt.grid(True)

        # Save plot to file
        plot_filename = os.path.join(output_path, f'{feature}_vs_price.png')
        plt.savefig(plot_filename)
        plt.close()


def preprocessing_baseline(X: pd.DataFrame, y: pd.Series):
    # Save the trip_id_unique_station column
    trip_id_unique_station = X["trip_id_unique_station"].copy()

    # Convert to datetime
    X = X.copy()  # To avoid SettingWithCopyWarning
    X.loc[:, 'door_closing_time'] = pd.to_datetime(X['door_closing_time'])
    X.loc[:, 'arrival_time'] = pd.to_datetime(X['arrival_time'])

    # Create door delta columns
    X["door_close_delta"] = None
    mask_notna = X["door_closing_time"].notna()
    X.loc[mask_notna, 'door_close_delta'] = (
            X.loc[mask_notna, 'door_closing_time'] - X.loc[mask_notna, 'arrival_time']
    ).dt.total_seconds()
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
    return mse_board


if __name__ == '__main__':
    # parser = ArgumentParser()
    # parser.add_argument('--training_set', type=str, required=True,
    #                     help="path to the training set")
    # parser.add_argument('--test_set', type=str, required=True,
    #                     help="path to the test set")
    # parser.add_argument('--out', type=str, required=True,
    #                     help="path of the output file as required in the task description")
    # args = parser.parse_args()

    # 1. load the training set (args.training_set)
    train_bus = pd.read_csv(TRAIN_BUS_CSV_PATH, encoding=ENCODER)
    x_passenger = pd.read_csv(X_PASSENGER, encoding=ENCODER)
    sample_size = 0.05  # 5% of the data
    baseline = train_bus.sample(frac=sample_size, random_state=RANDOM_STATE)
    remaining_data = train_bus.drop(baseline.index)
    x_base_line = baseline[x_passenger.columns]
    y_base_line = baseline["passengers_up"]

    # 2. preprocess the training set
    logging.info("preprocessing train...")
    X_train, X_test, y_train, y_test = preprocessing_baseline(x_base_line, y_base_line)

    # feature evaluation
    # feature_evaluation(X_train, y_train)

    # 3. train a model
    logging.info("training...")
    mse_linear = linear_regression(X_train, X_test, y_train, y_test)
    print('Decision linear_regression')
    print(f'Mean Squared Error: {mse_linear}')

    mse_trees = desition_trees(X_train, X_test, y_train, y_test)
    print('Decision Tree Regression')
    print(f'Mean Squared Error: {mse_trees}')

    mse_poly = polynomial_fitting(X_train, X_test, y_train, y_test)
    print('Decision polynomial_fitting')
    print(f'Mean Squared Error: {mse_poly}')

    # 4. load the test set (args.test_set)
    # 5. preprocess the test set
    logging.info("preprocessing test...")

    # 6. predict the test set using the trained model
    logging.info("predicting...")

    # 7. save the predictions to args.out
    # logging.info("predictions saved to {}".format(args.out))
