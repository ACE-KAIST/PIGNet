Code for the paper: "[PIGNET: A physics-informed deep learning model toward generalized drug-target interaction predictions](https://doi.org/10.1039/D1SC06946B)" by Seokhyun Moon, Wonho Zhung, Soojung Yang, Jaechang Lim, Woo Youn Kim

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

you can manually install the necessary packages for executing our code. The package list is shown as follows
- rdkit
- pytorch
- numpy
- biopython
- scikit-learn
- scipy
- smina

or you can just execute the followings in the command line.
```
./dependencies
conda activate pignet
```

## Running 
### Data
You can generate the data.  
- Training data by executing `data/download_train_data.sh`.  
- Benchmark data by executing `data/download_test_data.sh`.  

### Train
You should download and process the data with `data/download_test_data.sh`.  
Then, Trainig can be done with `run.sh`.  
This process will use the data in `./data/pdbbind_v2019/*`.  

### Model benchmark with CASF2016 benchmark and csar
You should download and process the data with `data/download_test_data.sh`.  
After that, you can use the execution commands for the benchmark are shown in `benchmarks/*/test.sh`.  

## Citing this work
~~~
@article{moon2022pignet,
  title={PIGNet: A physics-informed deep learning model toward generalized drug-target interaction predictions},
  author={Moon, Seokhyun and Zhung, Wonho and Yang, Soojung and Lim, Jaechang and Kim, Woo Youn},
  journal={Chemical Science},
  year={2022},
  publisher={Royal Society of Chemistry}
}
~~~
