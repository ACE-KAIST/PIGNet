#!/bin/bash
# CSAR1
python -u ../test.py \
--batch_size=64 \
--num_workers=0 \
--restart_file=../save/save_1000.pt \
--n_gnn=3 \
--dim_gnn=128 \
--test_result_filename=result_csar1_harmonic_1000 \
--ngpu=0 \
--interaction_net \
--potential="harmonic" \
--data_dir=../data/csar1/data/ \
--filename=../data/csar1/pdb_to_affinity.txt \
--key_dir=../data/csar1/keys/ \
> test_csar1_harmonic_1000

grep "R: " test_csar1_harmonic_1000

# CSAR2
python -u ../test.py \
--batch_size=64 \
--num_workers=0 \
--restart_file=../save/save_1000.pt \
--n_gnn=3 \
--dim_gnn=128 \
--test_result_filename=result_csar2_harmonic_1000 \
--ngpu=0 \
--interaction_net \
--potential="harmonic" \
--data_dir=../data/csar2/data/ \
--filename=../data/csar2/pdb_to_affinity.txt \
--key_dir=../data/csar2/keys/ \
> test_csar2_harmonic_1000

grep "R: " test_csar2_harmonic_1000

# casf2016_scroing
python -u ../test.py \
--batch_size=64 \
--num_workers=0 \
--restart_file=../save/save_1000.pt \
--n_gnn=3 \
--dim_gnn=128 \
--test_result_filename=result_scoring_harmonic_1000 \
--ngpu=0 \
--interaction_net \
--potential="harmonic" \
--data_dir=../data/CASF-2016/scoring/data/ \
--filename=../data/CASF-2016/scoring/pdb_to_affinity.txt \
--key_dir=../data/CASF-2016/scoring/keys/ \
> test_scoring_harmonic_1000

# Scoring
python ../casf2016_benchmark/scoring_power.py result_scoring_harmonic_1000 1000
# Ranking
python ../casf2016_benchmark/ranking_power.py result_scoring_harmonic_1000 1000

# casf2016_docking
python -u ../test.py \
--batch_size=64 \
--num_workers=0 \
--restart_file=../save/save_1000.pt \
--n_gnn=3 \
--dim_gnn=128 \
--test_result_filename=result_docking_harmonic_1000 \
--ngpu=1 \
--interaction_net \
--potential="harmonic" \
--data_dir=../data/CASF-2016/docking/data/ \
--filename=../data/CASF-2016/docking/pdb_to_affinity.txt \
--key_dir=../data/CASF-2016/docking/keys/ \
> test_docking_harmonic_1000

# Docking
python ../casf2016_benchmark/docking_power.py result_docking_harmonic_1000 1000

# casf2016_screening
# Execute below code changing the index of key_dir and std_out from 0 to 99
python -u ../test.py \
--batch_size=64 \
--num_workers=0 \
--restart_file=../save/save_1000.pt \
--n_gnn=3 \
--dim_gnn=128 \
--test_result_filename=result_screening_harmonic_1000 \
--ngpu=1 \
--interaction_net \
--potential="harmonic" \
--data_dir=../data/CASF-2016/docking/data/ \
--filename=../data/CASF-2016/docking/pdb_to_affinity.txt \
--key_dir=../data/CASF-2016/docking/keys/{0 to 99}/ \
> test_screening_harmonic_1000_{0 to 99}
cat result_screening_harmonic_1000_* > total_result.txt
python ../casf2016_benchmark/screening_power.py total_result.txt 1000
