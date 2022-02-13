#!/bin/bash
python -u ../../test.py \
--batch_size=64 \
--num_workers=4 \
--restart_file=../../save/PIGNet/save_2300.pt \
--n_gnn=3 \
--dim_gnn=128 \
--test_result_filename=result_csar1_pignet_2300 \
--ngpu=0 \
--interaction_net \
--model="pignet" \
--data_dir=../../data/csar1/data \
--filename=../../data/csar1/pdb_to_affinity.txt \
--key_dir=../../data/csar1/keys \
> test_csar1_pignet_2300
