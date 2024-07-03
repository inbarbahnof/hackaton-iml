from argparse import ArgumentParser
import pandas as pd
from sklearn.metrics import mean_squared_error


"""
usage:
    python evaluation_scripts/eval_importance_ratings.py --importance_ratings_predictions PATH --gold_importance_ratings PATH

example:
    python evaluation_scripts/eval_importance_ratings.py --importance_ratings_predictions predictions/importance_ratings_predictions.csv --gold_importance_ratings private/data/LINKED.HEB/y_importance_ratings.csv

"""


def eval_importance(predictions: pd.DataFrame, ground_truth: pd.DataFrame):
    combined = pd.merge(predictions, ground_truth, on='unique_id')
    mse_creativity = mean_squared_error(combined["creativity_important_x"], combined["creativity_important_y"])
    mse_ambition = mean_squared_error(combined["ambition_important_x"], combined["ambition_important_y"])
    mean_mse = (mse_creativity + mse_ambition) / 2
    return mean_mse


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--importance_ratings_predictions", type=str,
                        default="predictions/importance_ratings_predictions.csv",
                        help="path to the importance ratings predictions file")
    parser.add_argument("--gold_importance_ratings", type=str, default="private/data/y_importance_ratings.csv",
                        help="path to the gold passenger up file")
    args = parser.parse_args()

    importance_predictions = pd.read_csv(args.importance_ratings_predictions)
    gold_importance = pd.read_csv(args.gold_importance_ratings)
    mse_importance = eval_importance(importance_predictions, gold_importance)
    print(f"MSE for importance: {mse_importance}")
