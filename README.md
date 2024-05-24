## DPZero: Private Fine-Tuning of Language Models without Backpropagation

The code is for reproducing the results of DPZero on RoBERTa and OPT.
Our implementation is based on [MeZO](https://github.com/princeton-nlp/MeZO).

### Installation

This code is tested on `python 3.9.7`, with
`torch==2.0.1`, `transformers==4.28.1`, and `opacus==1.4.0`.

More on enviroments can be found in `environments.yml`. You can also create one using commands below.
```bash
conda env create -n dpzero -f environments.yml
conda activate dpzero
```

### Results


### Reproducing our results

The code for RoBERTa and OPT can be found in `./roberta` and `./opt`. 
Please see the detailed instruction therein.

