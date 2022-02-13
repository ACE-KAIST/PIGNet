#!/usr/bin/env python
import os
import pickle
from argparse import ArgumentParser, Namespace
from typing import List


def arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "-d",
        "--data_dir",
        help="data directory",
        type=str,
    )
    parser.add_argument(
        "-k",
        "--key_dir",
        help="key directory",
        type=str,
    )
    parser.add_argument(
        "-c",
        "--coreset_key_file",
        help="casf 2016 coreset keys from http://www.pdbbind.org.cn/casf.php",
        type=str,
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="generate train keys",
    )
    args, _ = parser.parse_known_args()
    return args


def write_keys(keys: List[str], file: str) -> None:
    with open(file, "wb") as w:
        pickle.dump(keys, w)
    return


def main(args: Namespace) -> None:
    keys = os.listdir(args.data_dir)
    with open(args.coreset_key_file, "r") as f:
        coreset_keys = [line.split()[0] for line in f.readlines()]

    train_keys = []
    test_keys = []
    for key in keys:
        if key.split("_")[0] in coreset_keys:
            test_keys.append(key)
        else:
            train_keys.append(key)

    if args.train:
        write_keys(train_keys, os.path.join(args.key_dir, "train_keys.pkl"))
    write_keys(test_keys, os.path.join(args.key_dir, "test_keys.pkl"))
    return


if __name__ == "__main__":
    args = arguments()
    if not os.path.exists(args.key_dir):
        os.mkdir(args.key_dir)
    main(args)
