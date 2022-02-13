#!/bin/bash
python -u ../../test.py \
--batch_size=64 \
--num_workers=4 \
--restart_file=../../save/PIGNet/save_2300.pt \
--n_gnn=3 \
--dim_gnn=128 \
--test_result_filename=result_screening_pignet_2300 \
--ngpu=1 \
--interaction_net \
--model="pignet" \
--data_dir=../../data/casf2016/screening/data \
--filename=../../data/casf2016/screening/pdb_to_affinity.txt \
--key_dir=../../data/casf2016/screening/keys \
> test_screening_pignet_2300
