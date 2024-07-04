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
        plot_filename = os.path.join(output_path, f'{feature}_vs_people_on_bus.png')
        plt.savefig(plot_filename)
        plt.close()


def preprocessing_baseline(X: pd.DataFrame, y: pd.Series):
    # Save the trip_id_unique_station column
    f_station_cnt =X.groupby("trip_id_unique")["trip_id_unique_station"].nunique().to_frame("station_cnt")
    f_total_passenger = X.groupby("trip_id_unique")["passengers_up"].sum().to_frame("total_passenger")
    f_mean_passenger = X.groupby("trip_id_unique")["passengers_up"].mean().to_frame("mean_passenger")
    f_mean_passenger_c = X.groupby("trip_id_unique")["passengers_continue"].mean().to_frame("mean_passenger_c")
    f_start_time = X.groupby("trip_id_unique")["arrival_time"].min().to_frame("start_time")
    f_start_time['start_time'] = pd.to_datetime(f_start_time['start_time']).dt.hour
    features = pd.concat([f_station_cnt,f_total_passenger, f_mean_passenger, f_mean_passenger_c, f_start_time], axis=1)
    features = features.merge(dur_baseline[["trip_id_unique","cluster","direction","mekadem_nipuach_luz"]],on = "trip_id_unique")
    label_encoder = LabelEncoder()
    features['cluster'] = label_encoder.fit_transform(features['cluster'])
    features = features.merge(y[["trip_id_unique","delta"]],on ="trip_id_unique" )
    features = features.drop_duplicates ()
    X_duration = features.drop(columns= ["delta","trip_id_unique"])
    Y_duration = features["delta"]
    X_train,X_test,y_train,y_test = sk.train_test_split(X_duration,Y_duration,train_size=0.75,random_state=RANDOM_STATE)
    X_train = X_train.set_index("trip_id_unique")
    X_test = X_test.set_index("trip_id_unique")
    return X_train, X_test, y_train, y_test


def linear_regression(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series,
                      y_test: pd.Series):
    # Ensure X_test has trip_id_unique_station column
    if 'trip_id_unique' not in X_test.columns:
        raise ValueError("X_test must contain 'trip_id_unique_station' column")

    # Drop the 'trip_id_unique_station' column from X_train and X_test
    X_train = X_train.drop(columns=['trip_id_unique'])
    X_test = X_test.drop(columns=['trip_id_unique'])

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
        'trip_id_unique': X_test.index,  # Ensure we have the correct index for trip_id_unique_station
        'dur_x': y_pred
    })

    ground_truth_df = pd.DataFrame({
        'trip_id_unique': X_test.index,  # Ensure we have the correct index for trip_id_unique_station
        'dur_y': y_test.values
    })

    # Evaluate using eval_boardings
    mse_board = eval.eval_boardings(predictions_df, ground_truth_df)
    return mse_board

def csv_output(mse:float ,passengers_up: pd.Series, trip_id_unique_station):
    # Create DataFrame with predictions
    predictions_df = pd.DataFrame({
        'trip_id_unique_station': trip_id_unique_station,
        'passengers_up': passengers_up.round()
    })

    # Save predictions to CSV file
    predictions_df.to_csv('passengers_up_predictions.csv', index=False)

def __creating_labels(dur_baseline):
    min_max_time = dur_baseline.groupby("trip_id_unique")["arrival_time"].agg({"min","max"}).reset_index()
    min_max_time["max"] = pd.to_datetime(min_max_time["max"])
    min_max_time["min"] = pd.to_datetime(min_max_time["min"])
    min_max_time["delta"] = (min_max_time["max"] - min_max_time["min"])/pd.Timedelta(1,"m")
    min_max_time["delta"] =round(min_max_time["delta"],2)
    return min_max_time[["trip_id_unique","delta"]]


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
    x_trip_duration = pd.read_csv(X_TRIP, encoding=ENCODER)
    lines_for_baseline = train_bus["trip_id_unique"].drop_duplicates().sample(frac = 0.10,random_state= RANDOM_STATE)
    dur_baseline  = train_bus[train_bus["trip_id_unique"].isin(lines_for_baseline)]
    dur_baseline = dur_baseline[x_trip_duration.columns]
    dur_labels = __creating_labels(dur_baseline)

    # 2. preprocess the training set
    logging.info("preprocessing train...")
    X_train, X_test, y_train, y_test = preprocessing_baseline(dur_baseline, dur_labels)

    # 3. train a model
    mse_poly = linear_regression(X_train, X_test, y_train, y_test)
    print('Decision polynomial_fitting')
    print(f'Mean Squared Error: {mse_poly}')

    # 4. load the test set (args.test_set)
    # 5. preprocess the test set
    logging.info("preprocessing test...")

    # 6. predict the test set using the trained model
    logging.info("predicting...")

    # 7. save the predictions to args.out
    # logging.info("predictions saved to {}".format(args.out))

