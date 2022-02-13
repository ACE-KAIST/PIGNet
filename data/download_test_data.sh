#!/bin/bash

SCORING_DIR=casf2016/scoring
DOCKING_DIR=casf2016/docking
SCREENING_DIR=casf2016/screening

mkdir -p $SCORING_DIR
mkdir -p $DOCKING_DIR
mkdir -p $SCREENING_DIR

# scoring
wget https://zenodo.org/record/6047984/files/casf2016_scoring.tar.gz?download=1 -O $SCORING_DIR/data.tar.gz
cd $SCORING_DIR
tar -xzf data.tar.gz
../../generate_keys.py -d data -k keys -c ../../coreset_keys.txt
../../pdb_to_affinity.py -d data -f pdb_to_affinity.txt -i ../../CoreSet.dat --benchmark
cd -
echo "Downloaded, unpacked and processed casf2016 scoring data."

# docking
wget https://zenodo.org/record/6047984/files/casf2016_docking.tar.gz?download=1 -O $DOCKING_DIR/data.tar.gz
cd $DOCKING_DIR
tar -xzf data.tar.gz
../../generate_keys.py -d data -k keys -c ../../coreset_keys.txt
../../pdb_to_affinity.py -d data -f pdb_to_affinity.txt -i ../../CoreSet.dat --benchmark
cd -
echo "Downloaded, unpacked and processed casf2016 docking data."

# screening
wget https://zenodo.org/record/6047984/files/casf2016_screening.tar.gz?download=1 -O $SCREENING_DIR/data.tar.gz
cd $SCREENING_DIR
tar -xzf data.tar.gz
../../generate_keys.py -d data -k keys -c ../../coreset_keys.txt
../../pdb_to_affinity.py -d data -f pdb_to_affinity.txt -i ../../CoreSet.dat --benchmark --screening
cd -
echo "Downloaded, unpacked and processed casf2016 screening data."
