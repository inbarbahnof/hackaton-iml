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
import evaluation_scripts.eval_trip_duration as eval
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

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
    f_station_cnt = X.groupby("trip_id_unique")["trip_id_unique_station"].nunique().to_frame(
        "station_cnt")
    f_total_passenger = X.groupby("trip_id_unique")["passengers_up"].sum().to_frame(
        "total_passenger")
    f_mean_passenger = X.groupby("trip_id_unique")["passengers_up"].mean().to_frame(
        "mean_passenger")
    f_mean_passenger_c = X.groupby("trip_id_unique")["passengers_continue"].mean().to_frame(
        "mean_passenger_c")
    f_start_time = X.groupby("trip_id_unique")["arrival_time"].min().to_frame("start_time")
    f_start_time['start_time'] = pd.to_datetime(f_start_time['start_time']).dt.hour
    features = pd.concat(
        [f_station_cnt, f_total_passenger, f_mean_passenger, f_mean_passenger_c, f_start_time],
        axis=1)
    features = features.merge(
        dur_baseline[["trip_id_unique", "cluster", "direction", "mekadem_nipuach_luz"]],
        on="trip_id_unique")
    label_encoder = LabelEncoder()
    features['cluster'] = label_encoder.fit_transform(features['cluster'])
    features = features.merge(y[["trip_id_unique", "delta"]], on="trip_id_unique")
    features = features.drop_duplicates()
    X_duration = features.drop(columns=["delta"])
    Y_duration = features["delta"]
    X_train, X_test, y_train, y_test = sk.train_test_split(X_duration, Y_duration, train_size=0.75,
                                                           random_state=RANDOM_STATE)
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
        'trip_id_unique': X_test.index,
        # Ensure we have the correct index for trip_id_unique_station
        'trip_duration_in_minutes_x': y_pred
    })

    ground_truth_df = pd.DataFrame({
        'trip_id_unique': X_test.index,
        # Ensure we have the correct index for trip_id_unique_station
        'trip_duration_in_minutes_y': y_test.values
    })

    # Evaluate using eval_boardings
    mse_board = eval.eval_duration(predictions_df, ground_truth_df)
    csv_output(y_pred, X_test.index)
    return mse_board


def csv_output(passengers_up: pd.Series, trip_id_unique_station):
    # Create DataFrame with predictions
    predictions_df = pd.DataFrame({
        'trip_id_unique': trip_id_unique_station,
        'trip_duration_in_minutes': passengers_up.round()
    })

    # Save predictions to CSV file
    predictions_df.to_csv('trip_duration_predictions.csv', index=False)


def __creating_labels(dur_baseline):
    min_max_time = dur_baseline.groupby("trip_id_unique")["arrival_time"].agg(
        {"min", "max"}).reset_index()
    min_max_time["max"] = pd.to_datetime(min_max_time["max"])
    min_max_time["min"] = pd.to_datetime(min_max_time["min"])
    min_max_time["delta"] = (min_max_time["max"] - min_max_time["min"]) / pd.Timedelta(1, "m")
    min_max_time["delta"] = round(min_max_time["delta"], 2)
    return min_max_time[["trip_id_unique", "delta"]]

def run_pca():
    # Load the data
    data = pd.read_csv('dur_baseline.csv')

    # Convert time-related columns to datetime
    data['arrival_time'] = pd.to_datetime(data['arrival_time'], format='%H:%M:%S', errors='coerce')
    data['door_closing_time'] = pd.to_datetime(data['door_closing_time'], format='%H:%M:%S', errors='coerce')

    # Calculate door closing delta
    data['door_close_delta'] = (data['door_closing_time'] - data['arrival_time']).dt.total_seconds()

    # Fill NaN values in door_close_delta with the mean
    data['door_close_delta'].fillna(data['door_close_delta'].mean(), inplace=True)

    # Drop original time columns
    data.drop(columns=['arrival_time', 'door_closing_time'], inplace=True)

    # Label encode categorical columns
    label_cols = ['part', 'trip_id_unique_station', 'trip_id_unique', 'line_id', 'direction', 'alternative', 'cluster', 'station_name']
    label_encoder = LabelEncoder()

    for col in label_cols:
        data[col] = label_encoder.fit_transform(data[col])

    # Ensure all columns are numeric
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # Handle any remaining NaN values
    data.fillna(data.mean(), inplace=True)

    # Drop non-feature columns if any
    data.drop(columns=['trip_id'], inplace=True)

    # Separate features and target variable if needed
    X = data.drop(columns=['passengers_up'])
    y = data['passengers_up']

    # Normalize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)

    # Plot the explained variance ratio
    plt.figure(figsize=(10, 6))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--')
    plt.title('Explained Variance by Principal Components')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.grid()
    plt.show()

    # Select the number of components that explain a good amount of variance, e.g., 95%
    explained_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(explained_variance >= 0.95) + 1

    print(f'Number of components that explain 95% of the variance: {n_components}')

    # Apply PCA with the selected number of components
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X_scaled)

    # Convert to DataFrame
    X_reduced_df = pd.DataFrame(X_reduced, columns=[f'PC{i+1}' for i in range(n_components)])

    # Optionally, add the target variable back to the DataFrame
    X_reduced_df['passengers_up'] = y.values

    # Display the first few rows of the reduced DataFrame
    # print(X_reduced_df.head())

    return X_reduced_df

 
    

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
    lines_for_baseline = train_bus["trip_id_unique"].drop_duplicates().sample(frac=0.10,
                                                                              random_state=RANDOM_STATE)
    dur_baseline = train_bus[train_bus["trip_id_unique"].isin(lines_for_baseline)]
    # dur_baseline.to_csv("dur_baseline.csv")
    dur_baseline = dur_baseline[x_trip_duration.columns]
    dur_labels = __creating_labels(dur_baseline)

    # 2. preprocess the training set
    logging.info("preprocessing train...")
    X_train, X_test, y_train, y_test = preprocessing_baseline(dur_baseline, dur_labels)

    # 3. train a model
    mse_poly = linear_regression(X_train, X_test, y_train, y_test)
    print('Decision linear_regression')
    print(f'Mean Squared Error: {mse_poly}')

    X_reduced = run_pca()
    dur_baseline = X_reduced.drop(columns = ["PC12","PC13","PC14","passengers_up"])
    dur_labels = X_reduced["passengers_up"]

    # 2. preprocess the training set
    # X_train, X_test, y_train, y_test = preprocessing_baseline(dur_baseline, dur_labels)
    X_train, X_test, y_train, y_test  = sk.train_test_split(dur_baseline,dur_labels,train_size=0.75,random_state=RANDOM_STATE)
        # Linear regression
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)
    # mse_board = eval.eval_duration(predictions_df, ground_truth_df)
    # mse_poly = mean_squared_error(dur_labels, y_pred)
    #mse_poly = linear_regression(X_train, X_test, y_train, y_test)
    print('Decision polynomial_fitting after pca')
    print(f'Mean Squared Error: {mse_poly}')

    # 4. load the test set (args.test_set)
    # 5. preprocess the test set
    logging.info("preprocessing test...")

    # 6. predict the test set using the trained model
    logging.info("predicting...")

    # 7. save the predictions to args.out
    # logging.info("predictions saved to {}".format(args.out))
