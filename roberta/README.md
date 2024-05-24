## DPZero on RoBERTa

The code is for reproducing the results of DPZero on RoBERTa-large. 
Our implementation is based on [MeZO](https://github.com/princeton-nlp/MeZO).

### Preparations

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
