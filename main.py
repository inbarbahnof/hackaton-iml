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
    # creating door delta columns
    x_base_line['door_closing_time'] = pd.to_datetime(x_base_line['door_closing_time'])
    x_base_line['arrival_time'] = pd.to_datetime(x_base_line['arrival_time'])
    x_base_line["door_close_delta"] = None
    x_base_line.loc[x_base_line["door_closing_time"].notna(), ['door_close_delta']] = (
            x_base_line.loc[x_base_line["door_closing_time"].notna(), 'door_closing_time'] -
            x_base_line.loc[
                x_base_line["door_closing_time"].notna(), 'arrival_time']).dt.total_seconds()
    door_delta_mean = x_base_line["door_close_delta"].mean()
    x_base_line["door_close_delta"] = x_base_line["door_close_delta"].fillna(door_delta_mean)
    x_base_line['arrival_time'] = pd.to_datetime(x_base_line['arrival_time'])

    # catgorized arrival time
    arrival_hours = x_base_line['arrival_time'].dt.hour
    percentiles = arrival_hours.describe(percentiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    percentile_values = percentiles.loc[
        ['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%']].values
    labels = [f'{int(value)}' for value in percentile_values]
    labels.insert(0, '0')
    x_base_line['arrival_time_label'] = pd.cut(arrival_hours,
                                               bins=[0] + list(percentile_values) + [24],
                                               labels=labels,
                                               include_lowest=True)
    # Label Encoding
    label_encoder = LabelEncoder()
    x_base_line['part_encoded'] = label_encoder.fit_transform(x_base_line['part'])
    x_base_line['alternative_encoded'] = label_encoder.fit_transform(x_base_line['alternative'])
    del x_base_line["arrival_time"]
    del x_base_line["door_closing_time"]
    del x_base_line["cluster"]
    del x_base_line["station_name"]
    del x_base_line["part"]
    del x_base_line["trip_id_unique"]
    del x_base_line["trip_id_unique_station"]
    del x_base_line["alternative"]

    return sk.train_test_split(x_base_line, y_base_line, test_size=0.25,
                               random_state=RANDOM_STATE)


def desition_trees(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series,
                   y_test: pd.Series):
    # descition trees
    model_dt = DecisionTreeRegressor(random_state=RANDOM_STATE)
    model_dt.fit(X_train, y_train)
    # Predict on the test set
    y_pred_dt = model_dt.predict(X_test)
    # Calculate performance metrics
    mse_dt = mean_squared_error(y_test, y_pred_dt)
    r2_dt = r2_score(y_test, y_pred_dt)
    return mse_dt, r2_dt


def linear_regression(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series,
                      y_test: pd.Series):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2


def polynomial_fitting(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series,
                       y_test: pd.Series):
    # polynomial fiiting
    poly = PolynomialFeatures(degree=2)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    # Initialize and train the Polynomial Regression model
    model_poly = LinearRegression()
    model_poly.fit(X_train_poly, y_train)
    # Predict on the test set
    y_pred_poly = model_poly.predict(X_test_poly)
    # Calculate MSE
    mse_poly = mean_squared_error(y_test, y_pred_poly)
    return mse_poly


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

    #feature evaluation
    # feature_evaluation(X_train, y_train)

    # 3. train a model
    logging.info("training...")
    mse_linear, r2_linear = linear_regression(X_train, X_test, y_train, y_test)
    mse_trees, r2_trees = desition_trees(X_train, X_test, y_train, y_test)
    mse_poly = polynomial_fitting(X_train, X_test, y_train, y_test)

    print('Decision linear_regression')
    print(f'Mean Squared Error: {mse_linear}')

    print('Decision Tree Regression')
    print(f'Mean Squared Error: {mse_trees}')

    print('Decision polynomial_fitting')
    print(f'Mean Squared Error: {mse_poly}')

    # 4. load the test set (args.test_set)
    # 5. preprocess the test set
    logging.info("preprocessing test...")

    # 6. predict the test set using the trained model
    logging.info("predicting...")

    # 7. save the predictions to args.out
    # logging.info("predictions saved to {}".format(args.out))
