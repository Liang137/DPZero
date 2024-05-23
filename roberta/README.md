## DPZero on RoBERTa-large

The code is for reproducing the results of DPZero on RoBERTa-large. 
Our implementation is based on [MeZO](https://github.com/princeton-nlp/MeZO).

### Installation

This code is tested on `Python 3.9.7`, with
`torch==2.0.1`, `transformers==4.28.1`, and `opacus==1.4.0`.

More on enviroments can be found in `enviroments.yml`. You can also create one using commands below.
```bash
conda env create -n dpzero -f enviroments.yml
conda activate dpzero
```

### Prepare the data

The datasets can be found [here](https://nlp.cs.princeton.edu/projects/lm-bff/datasets.tar). 
Please download it and extract the files to `./data/original`, or run the following commands:

```bash
cd data
bash prepare_datasets.sh
cd ..
```
### Examples

To reproduce the results (on dataset SST-2) of our paper, 
you can run them directly with the following commands. 
Results on other datasets can be obtained by changing `TASK` to `sst-5`, `SNLI`, `MNLI`, `RTE`, and `trec`. 
```bash
# DPZero
bash examples/dpzero.sh

# Baseline for non DP training with zeroth order method
bash examples/mezo.sh
```
