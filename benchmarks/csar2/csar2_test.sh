#!/bin/bash
python -u ../../test.py \
--batch_size=64 \
--num_workers=0 \
--restart_file=../../save/save_1000.pt \
--n_gnn=3 \
--dim_gnn=128 \
--test_result_filename=result_csar2_harmonic_1000 \
--ngpu=0 \
--interaction_net \
--potential="harmonic" \
--data_dir=../../data/csar2/data \
--filename=../../data/csar2/pdb_to_affinity.txt \
--key_dir=../../data/csar2/keys \
> test_csar2_harmonic_1000
