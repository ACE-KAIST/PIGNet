#!/usr/bin/env python
import argparse
import glob
from typing import List

import numpy as np
from scipy import stats


def bootstrap_confidence(
    values: List[float], n: int = 10000, confidence: float = 0.9
) -> np.ndarray:
    metrics = []
    for _ in range(n):
        indice = np.random.randint(0, len(values), len(values))
        sampled = [values[i] for i in indice]
        metrics.append(sum(sampled) / len(sampled))
    metrics = np.array(metrics)
    return stats.t.interval(
        confidence, len(metrics) - 1, loc=np.mean(metrics), scale=np.std(metrics)
    )


def predictive_index(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    n = len(pred)
    ws, cs = [], []
    for i in range(n):
        for j in range(i + 1, n):
            w = abs(true[j] - true[i])
            c = -1
            if (pred[j] - pred[i]) * (true[j] - true[i]) > 0:
                c = 1
            elif true[j] - true[i] == 0:
                c = 0
            ws.append(w)
            cs.append(c)
    ws = np.array(ws)
    cs = np.array(cs)
    return np.sum(ws * cs) / np.sum(ws)


def main(args: argparse.Namespace) -> None:
    with open(args.ranking_file, "r") as f:
        lines = f.readlines()[1:]
        pdbs = [line.split()[0].lower() for line in lines]
        clusters = [pdbs[i * 5 : i * 5 + 5] for i in range(57)]
        pdb_to_true = {line.split()[0]: float(line.split()[3]) for line in lines}

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
        with open(file) as f:
            lines = f.readlines()
            lines = [line.split() for line in lines]
            pdb_to_pred = dict({line[0]: float(line[2]) / -1.36 for line in lines})

        average_s_r, average_t_r = [], []
        for cluster in clusters:
            no_data = False
            for pdb in cluster:
                if pdb not in pdb_to_pred.keys():
                    no_data = True
            if no_data:
                continue

            preds = [pdb_to_pred[pdb] for pdb in cluster]
            preds, ordered_pdb = zip(*sorted(zip(preds, cluster)))
            true_order = [1, 2, 3, 4, 5]
            pred_order = [ordered_pdb.index(pdb) + 1 for pdb in cluster]
            s_r, _ = stats.spearmanr(true_order, pred_order)
            t_r, _ = stats.kendalltau(true_order, pred_order)
            average_s_r.append(s_r)
            average_t_r.append(t_r)
        confidence_interval = bootstrap_confidence(average_s_r, args.n_bootstrap)
        average_s_r = sum(average_s_r) / len(average_s_r)
        average_t_r = sum(average_t_r) / len(average_t_r)
        pi = predictive_index(
            [pdb_to_pred[key] for key in pdb_to_pred.keys()],
            [pdb_to_true[key] for key in pdb_to_pred.keys()],
        )
        if args.verbose:
            print(file, end="\t")
            print(round(average_s_r, 3), end="\t")
            print(round(average_t_r, 3), end="\t")
            print(round(confidence_interval[0], 3), end="\t")
            print(round(confidence_interval[1], 3))
        else:
            print(round(average_s_r, 3))
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ranking_file",
        type=str,
        default="./CoreSet.dat",
        help="CASF-2016 ranking dat file",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-f", "--files", type=str, default="result_scoring_")
    parser.add_argument("-n", "--n_bootstrap", type=int, default=100)
    args = parser.parse_args()

    main(args)
