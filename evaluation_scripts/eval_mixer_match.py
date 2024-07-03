from argparse import ArgumentParser
import pandas as pd
from sklearn.metrics import f1_score


"""
usage:
    python evaluation_scripts/eval_mixer_match.py --match_predictions PATH --gold_match PATH

example:
    python evaluation_scripts/eval_mixer_match.py --match_predictions predictions/match_predictions.csv --gold_match private/data/LINKED.HEB/y_match.csv

"""


def eval_match(predictions: pd.DataFrame, ground_truth: pd.DataFrame):
    combined = pd.merge(predictions, ground_truth, on='unique_id')
    f1_match = f1_score(combined["match_x"], combined["match_y"])
    return f1_match


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--match_predictions", type=str, default="predictions/match_predictions.csv",
                        help="path to the matching predictions file")
    parser.add_argument("--gold_match", type=str, default="private/data/y_match.csv",
                        help="path to the gold passenger up file")
    args = parser.parse_args()

    match_predictions = pd.read_csv(args.match_predictions)
    gold_match = pd.read_csv(args.gold_match)
    f1_match_score = eval_match(match_predictions, gold_match)
    print(f"F1 for match: {round(f1_match_score, 2)}")
