#!/bin/bash
python -u ../../test.py \
--batch_size=64 \
--num_workers=4 \
--restart_file=../../save/save_2300.pt \
--n_gnn=3 \
--dim_gnn=128 \
--test_result_filename=result_csar2_pignet_2300 \
--ngpu=0 \
--interaction_net \
--model="pignet" \
--data_dir=../../data/csar2/data \
--filename=../../data/csar2/pdb_to_affinity.txt \
--key_dir=../../data/csar2/keys \
> test_csar2_pignet_2300
