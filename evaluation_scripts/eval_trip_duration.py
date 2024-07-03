from argparse import ArgumentParser
import pandas as pd
from sklearn.metrics import mean_squared_error


"""
usage:
    python evaluation_scripts/eval_trip_duration.py --trip_duration_predictions PATH --gold_trip_duration PATH

example:
    python evaluation_scripts/eval_trip_duration.py --trip_duration_predictions predictions/trip_duration_predictions.csv --gold_trip_duration private/HU.BER/y_trip_duration.csv

"""


def eval_duration(predictions: pd.DataFrame, ground_truth: pd.DataFrame):
    combined = pd.merge(predictions, ground_truth, on='trip_id_unique')
    mse_loss = mean_squared_error(combined["trip_duration_in_minutes_x"], combined["trip_duration_in_minutes_y"])
    return mse_loss


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("--trip_duration_predictions", type=str, default="predictions/trip_duration_predictions.csv",
                        help="path to the trip duration predictions file")
    parser.add_argument("--gold_trip_duration", type=str, default="private/data/y_trip_duration.csv",
                        help="path to the gold trip duration file")
    args = parser.parse_args()

    duration_predictions = pd.read_csv(args.trip_duration_predictions, encoding="ISO-8859-8")
    gold_duration = pd.read_csv(args.gold_trip_duration, encoding="ISO-8859-8")
    mse = eval_duration(duration_predictions, gold_duration)
    print(f"MSE for duration: {mse}")
