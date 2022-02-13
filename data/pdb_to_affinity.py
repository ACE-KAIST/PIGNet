#!/usr/bin/env python
import os
from argparse import ArgumentParser, Namespace
from typing import Dict, List


def arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "-d",
        "--data_dir",
        help="data directory",
        type=str,
    )
    parser.add_argument(
        "-f",
        "--pdb_to_affinity_file",
        help="pdb to affinity file",
        type=str,
    )
    parser.add_argument(
        "-i",
        "--index_file",
        help="pdbbind or casf corset index file",
        type=str,
    )
    parser.add_argument(
        "--screening",
        action="store_true",
        help="screening data",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="benchmark data",
    )
    args, _ = parser.parse_known_args()
    return args


def parse_pdb_to_affinity(file: str, benchmark: bool = True) -> Dict[str, float]:
    """for scoring and docking"""
    start = 1 if benchmark else 6
    with open(file, "r") as f:
        lines = [line.split() for line in f.readlines()[start:]]
        lines = [[line[0], float(line[3])] for line in lines]
    return dict(lines)


def main(args: Namespace) -> None:
    keys = os.listdir(args.data_dir)

    if args.screening:
        affinity_dic = {key: 5 for key in keys}
    else:
        pdb_to_affinity = parse_pdb_to_affinity(args.index_file, args.benchmark)
        affinity_dic = {key: pdb_to_affinity[key.split("_")[0]] for key in keys}

    with open(args.pdb_to_affinity_file, "w") as w:
        for key, value in affinity_dic.items():
            w.write(f"{key}\t{value}\n")
    return


if __name__ == "__main__":
    args = arguments()
    dirname = os.path.dirname(args.pdb_to_affinity_file)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)
    main(args)
