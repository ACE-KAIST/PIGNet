#!/bin/bash

SCORING_DIR=pdbbind_v2019/scoring
DOCKING_DIR=pdbbind_v2019/docking
RANDOM_DIR=pdbbind_v2019/random
CROSS_DIR=pdbbind_v2019/cross

mkdir -p $SCORING_DIR
mkdir -p $DOCKING_DIR
mkdir -p $RANDOM_DIR
mkdir -p $CROSS_DIR

# # scoring
# wget https://zenodo.org/record/6047984/files/pdbbind_v2019_refined.tar.gz?download=1 -O $SCORING_DIR/data.tar.gz
# cd $SCORING_DIR
# tar -xzf data.tar.gz
# ../../generate_keys.py -d data -k keys -c ../../coreset_keys.txt --train
# ../../pdb_to_affinity.py -d data -f pdb_to_affinity.txt -i ../../INDEX_refined_data.2019
# cd -
# echo "Downloaded, unpacked and processed pdbbind v2019 scoring data."
# 
# # docking
# wget https://zenodo.org/record/6047984/files/pdbbind_v2019_docking.tar.gz?download=1 -O $DOCKING_DIR/data.tar.gz
# cd $DOCKING_DIR
# tar -xzf data.tar.gz
# ../../generate_keys.py -d data -k keys -c ../../coreset_keys.txt --train
# ../../pdb_to_affinity.py -d data -f pdb_to_affinity.txt -i ../../INDEX_refined_data.2019
# cd -
# echo "Downloaded, unpacked and processed pdbbind v2019 docking data."

# random screening
wget https://zenodo.org/record/6047984/files/pdbbind_v2019_random_screening.tar.gz?download=1 -O $RANDOM_DIR/data.tar.gz
cd $RANDOM_DIR
tar -xzf data.tar.gz
../../generate_keys.py -d data -k keys -c ../../coreset_keys.txt --train
../../pdb_to_affinity.py -d data -f pdb_to_affinity.txt -i ../../INDEX_refined_data.2019 --screening
cd -
echo "Downloaded, unpacked and processed pdbbind v2019 random screening data."

# cross screening
wget https://zenodo.org/record/6047984/files/pdbbind_v2019_cross_screening.tar.gz?download=1 -O $CROSS_DIR/data.tar.gz
cd $CROSS_DIR
tar -xzf data.tar.gz
../../generate_keys.py -d data -k keys -c ../../coreset_keys.txt --train
../../pdb_to_affinity.py -d data -f pdb_to_affinity.txt -i ../../INDEX_refined_data.2019 --screening
cd -
echo "Downloaded, unpacked and processed pdbbind v2019 random screening data."
