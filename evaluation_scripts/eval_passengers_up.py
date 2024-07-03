from argparse import ArgumentParser
import pandas as pd
from sklearn.metrics import mean_squared_error


"""
usage:
    python evaluation_scripts/eval_passengers_up.py --passengers_up_predictions PATH --gold_passengers_up PATH

example:
    python evaluation_scripts/eval_passengers_up.py --passengers_up_predictions predictions/passengers_up_predictions.csv --gold_passengers_up private/data/HU.BER/y_passengers_up.csv

"""


def eval_boardings(predictions: pd.DataFrame, ground_truth: pd.DataFrame):
    combined = pd.merge(predictions, ground_truth, on='trip_id_unique_station')
    mse_board = mean_squared_error(combined["passengers_up_x"], combined["passengers_up_y"])
    return mse_board


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("--passengers_up_predictions", type=str, default="predictions/passengers_up_predictions.csv",
                        help="path to the passenger up predictions file")
    parser.add_argument("--gold_passengers_up", type=str, default="private/data/y_passengers_up.csv",
                        help="path to the gold passenger up file")
    args = parser.parse_args()

    passenger_predictions = pd.read_csv(args.passengers_up_predictions, encoding="ISO-8859-8")
    gold_passenger_up = pd.read_csv(args.gold_passengers_up, encoding="ISO-8859-8")
    mse_boarding = eval_boardings(passenger_predictions, gold_passenger_up)
    print(f"MSE for boardings: {mse_boarding}")
