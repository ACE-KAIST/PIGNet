#!/bin/bash
python -u ../../test.py \
--batch_size=64 \
--num_workers=0 \
--restart_file=../../save/save_1000.pt \
--n_gnn=3 \
--dim_gnn=128 \
--test_result_filename=result_docking_harmonic_1000 \
--ngpu=1 \
--interaction_net \
--potential="harmonic" \
--data_dir=../../data/CASF-2016/docking/data \
--filename=../../data/CASF-2016/docking/pdb_to_affinity.txt \
--key_dir=../../data/CASF-2016/docking/keys \
> test_docking_harmonic_1000
