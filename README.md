Code for the paper: "[PIGNET: A physics-informed deep learning model toward generalized drug-target interaction predictions](https://arxiv.org/abs/2008.12249)" by Seokhyun Moon, Wonho Zhung, Soojung Yang, Jaechang Lim, Woo Youn Kim

# Table of Contents
- [Install Dependencies](#install-dependencies)
- [Running](#running)
    - [Data Preprocessing](#data-preprocessing)
    - [Train](#train)
    - [Model benchmark with CASF2016 benchmark and csar](#model-benchmark-with-casf2016-benchmark-and-csar)
- [Command Examples](#command-examples)
- [Citing this work](#citing-this-work)

## Install Dependencies
PIGNet needs conda environment. After installing [conda](https://www.anaconda.com/),

you can manually install the packages for our code. The package list is shown as follows
- rdkit
- pytorch
- numpy
- biopython
- ase
- anaconda scikit-learn
- scipy
- smina

or you can just execute the followings in the command line.
```
./dependencies
conda activate pignet
```

## Running 
### Data Preprocessing
To prepare our dataset for train and test, change directory to [data](https://github.com/jaechanglim/DTI_PDBbind/tree/master/data), and follow the instructions.

### Train
> **Important**: Default values of the rest of the arguments are set as the best parameters, which generated our results in the paper. Also, set `interaction_net` argument as `True`. It will affect the model enormously.

PIGNet uses four dataset: original(`data_dir`), docking(`data_dir2`), random\_screening(`data_dir3`), and cross\_screening(`data_dir4`), and each dataset has three arguments for train.
- `data_dir`: Directory consists of the preprocessed pickle data.
- `key_dir`: Directory consists of the keys for pickle data.
- `filename`: A path of the text file which consists of complex key and its binding affinity.

During the training, the results will be written at following arguments:
- `train_result_filename`
- `test_result_filename`
- `train_result_docking_filename`
- `test_result_docking_filename`
- `train_result_screening_filename`
- `test_result_screening_filename`

To train with the uncertainty, add the following arguments:
- `dropout_rate`: Set this argument as 0.2 unlike the train without uncertainty which `dropout_rate` is 0.1.
- `with_uncertainty`: Flag of using uncertainty during train.
- `mc_dropout`: Should set as True to use Monte-Carlo dropout.

We also realized 3D CNN model of [KDEEP](https://pubs.acs.org/doi/10.1021/acs.jcim.7b00650). To train 3D CNN model, you can use following arguments with the default values:
- `potential`: Select a model to train. Default value is "harmonic".
- `grid_rotatin`: Whether rotate the grid or not during the train.
- `lattice_dim`: Size of the 3D lattice.
- `scaling`: Interval of the lattice points.

### Model benchmark with CASF2016 benchmark and csar
> **Important**: To benchmark the model with CASF2016 benchmark, you should prepare `CASF-2016`, `csar1` and `csar2` data in [data](https://github.com/jaechanglim/DTI_PDBbind/tree/master/data) directory first.

Inside the `benchmarks` directory, execute `../test.py` with the corresponding test datasets. We did several benchmark study with CASF-2016 and csar datasets.
To test the model with specific dataset, give three arguments for test dataset.
- `data_dir`: Directory consists of the preprocessed pickle data.
- `key_dir`: Directory consists of the keys for pickle data.
- `filename`: A path of the text file which consists of complex key and its binding affinity.

The test result will be written at following arguments:
- `test_result_filename`

To test the model with uncertainty, add the following arguments:
- `with_uncertainty`: Flag of model trained with or without uncertainty.
- `n_mc_sampling`: Number of the Monte-Carlo sampling.


> **Important:** For different test dataset, executing `../test.py` code will generate several different result files. Followings are the ways to get the score for each benchmark. Execute the commands at the `benchmarks` directory. To get the *R score*, you should forward the std output to `output_file` for each benchmark.

* Csar1

You can just execute `csar1_test.sh` at the `benchmarks/csar1` directory. To get the R value from the forwarded `csar1_output_file`, just execute following:
```
grep 'R:' {csar1_output_file}*
```

* Csar2

You can just execute `csar2_test.sh` at the `benchmarks/csar2` directory. To get the R value from the forwarded `csar2_output_file`, just execute following:
```
grep 'R:' {csar2_output_file}*
```

> **Important:** To test our save file(`save/save_1000.pt`), you should set `{epoch}` as 1000.

* Scoring power

You can just execute `scoring_test.sh` at the `benchmarks/scoring` directory. Then, execute the following command, where `scoring_result_file` is argument value of `test_result_filename` of `test.py`.
```
python ../../casf2016_benchmark/scoring_power.py {scoring_result_file} {epoch}
```

* Ranking power

Ranking power uses `scoring_result_file`. Just execute the following command, where `scoring_result_file` is argument value of `test_result_filename` of `test.py`.
```
python ../../casf2016_benchmark/ranking_power.py {scoring_result_file} {epoch}
```
* docking power

You can just execute `docking_test.sh` at the `benchmarks/docking` directory. Then, execute the following command, where `docking_result_file` is argument value of `test_result_filename` of `test.py`.
```
python ../../casf2016_benchmark/docking_power.py {docking_result_file} {epoch}
```
* screening power

To compute the screening power, you should iterate keys from 0 to 99 as shown in `benchmarks/screening/screening_test.sh`. Then, execute the following commands inside `benchmarks/screening` directory.
```
cat result_* > total_result.txt
python ../../casf2016_benchmark/screening_power.py total_result.txt {epoch}
```

## Command Examples
`train.sh` in this directory is the code that we used for training the model.
> Train
~~~
python -u train.py \
--save_dir=save \
--tensorboard_dir=run \
--train_result_filename=result_train.txt \
--test_result_filename=result_test.txt \
--train_result_docking_filename=result_train_docking.txt \
--test_result_docking_filename=result_test_docking.txt \
--train_result_screening_filename=result_train_screening.txt \
--test_result_screening_filename=result_test_screening.txt \
--data_dir={original data dir} \
--filename={original data file path} \
--key_dir={original data key dir} \
--data_dir2={docking data dir} \
--filename2={docking data file path} \
--key_dir2={docking data key dir} \
--data_dir3={random_screening data dir} \
--filename3={random_screening data file path} \
--key_dir3={random_screening data key dir} \
--data_dir4={cross_screening data dir} \
--filename4={cross_screening data file path} \
--key_dir4={cross_screening data key dir} \
--potential='harmonic' \
--interaction_net \
> output 2> /dev/null 
~~~
> Train With Uncertainty (Just add to train command)
~~~
--dropout_rate=0.2
--train_with_uncertainty
--mc_dropout=True
~~~

> Test

`test.sh` in `benchmarks` directory is the basic code that we used for training the model.

We use `casf2016_scoring`, `casf2016_ranking`, `casf2016_docking`, `casf2016_screening`, `csar1`, `csar2` to benchmark the model.
Use different `{benchmark name}`, `{benchmark data dir}`, `{benchmark data file path}`, and `{benchmark data key dir}` for each benchmark, in the following code example.

Also, for the `casf2016_docking` and `casf2016_screening`, we recommend to use `ngpu=1` option.
~~~
python -u ../test.py \
--batch_size=64 \
--num_workers=0 \
--restart_file=../save/save_1000.pt \
--n_gnn=3 \
--dim_gnn=128 \
--test_result_filename=result_{benchmark name}_harmonic_1000 \
--ngpu=0 \
--interaction_net \
--potential="harmonic" \
--data_dir={benchmark data dir} \
--filename={benchmark data file path} \
--key_dir={benchmark data key dir} \
> test_{benchmark name}_harmonic_1000
~~~

## Citing this work
~~~
@article{moon2020pignet,
    title={PIGNet: A physics-informed deep learning model toward generalized drug-target interaction predictions},
    author={Moon, Seokhyun and Zhung, Wonho and Yang, Soojung and Lim, Jaechang and Kim, Woo Youn},
    journal={arXiv preprint arXiv:2008.12249},
    year={2020}
}
~~~
