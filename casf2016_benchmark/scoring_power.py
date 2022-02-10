#!/usr/bin/env python
import argparse
import glob
from typing import List

import numpy as np
from scipy import stats


def mean_confidence_interval(
    data: List[float], confidence: float = 0.95
) -> List[float]:
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return m, m - h, m + h


def bootstrap_confidence(
    true: np.ndarray,
    pred: np.ndarray,
    n: int = 10000,
    confidence: float = 0.9,
) -> np.ndarray:
    Rs = []
    for _ in range(n):
        indice = np.random.randint(0, len(pred), len(pred))
        t = [true[i] for i in indice]
        p = [pred[i] for i in indice]
        a, b, R, _, std_err = stats.linregress(t, p)
        Rs.append(R)
    Rs = np.array(Rs)
    return stats.t.interval(confidence, len(Rs) - 1, loc=np.mean(Rs), scale=np.std(Rs))


def main(args: argparse.Namespace) -> None:
    files = glob.glob(args.files + "*")
    try:
        if "txt" in files[0]:
            files = sorted(
                files, key=lambda file: int(file.split("_")[-1].split(".")[0])
            )
        else:
            files = sorted(files, key=lambda file: int(file.split("_")[-1]))
    except Exception:
        pass
    for file in files:
        with open(file, "r") as f:
            lines = f.readlines()
        lines = [line.split() for line in lines]
        true = np.array([float(line[1]) for line in lines])
        pred = np.array([float(line[2]) for line in lines])
        a, b, R, _, std_err = stats.linregress(true, pred)
        fit_pred = a * pred + b
        SD = np.power(np.power(true - fit_pred, 2).sum() / (len(true) - 1), 0.5)
        confidence_interval = bootstrap_confidence(true, pred, args.n_bootstrap)
        if args.verbose:
            print(file, end="\t")
            print(round(R, 3), end="\t")
            print(round(SD, 3), end="\t")
            print(round(confidence_interval[0], 3), end="\t")
            print(round(confidence_interval[1], 3))
        else:
            print(round(R, 3))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-f", "--files", type=str, default="result_scoring_")
    parser.add_argument("-n", "--n_bootstrap", type=int, default=100)
    args = parser.parse_args()

    main(args)
