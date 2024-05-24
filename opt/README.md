## DPZero on OPT

This part of the code is for DPZero experiments on OPT (an autoregressive LM). 
It includes training autoregressive LMs with linear probing, head tuning, full fine-tuning, parameter-efficient fine-tuning (PEFT), MeZO, and DPZero. 
It also covers zero-shot and in-context learning (ICL) evaluation. It is tested on OPT-1.3B, 2.7B, and 6.7B, but should be able to extend to other sizes and other autoregressive LMs.


### Usage

Use `run.py` for all functions (zero-shot/ICL/fine-tuning/MeZO/DPZero):

```bash
python run.py {ARGUMENTS}
```

Please read `run.py` for a complete list of arguments. We introduce some of the most important ones below.
* `--model_name`: HuggingFace model name or path.
* `--task_name`: Task name.
* `--trainer`: can be `none` (zero-shot/ICL), `regular` (fine-tuning), or `zo` (DPZero/MeZO).
* `--train_as_classification`: turn this on for classification tasks (Cross Entropy over likelihood of each class' label words). Otherwise it is LM-style teacher forcing.
* `--zo_eps`: MeZO hyperparameter epsilon.
* `--dp_epsilon`: epsilon for differentially private training 
* `--dp_epsilon`: epsilon for differentially private training 
* `--dpzero`: Apply DPZero for memory efficient differentially private finetuning.
* `--dpzero_clip_threshold`: clipping threshold for DPZero.


We also support all [HuggingFace trainer arguments](https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py) for easily setting fine-tuning hyperparameters.

### Examples
We provide example scripts below for reproducing our experiments. All our examples sample 1,000 training examples, 500 validation examples, and 1,000 testing examples. For ICL, we use 32 demonstrations. For detailed hyperparameters and grid search configs, please refer to Appendix D of [our paper](https://arxiv.org/pdf/2305.17333.pdf).
```bash
# Zero-shot
MODEL=facebook/opt-1.3b TASK=SST2 bash examples/icl.sh --num_train 0

# In-context learning
MODEL=facebook/opt-1.3b TASK=SST2 bash examples/icl.sh 

# Full-parameter fine-tuning
MODEL=facebook/opt-1.3b TASK=SST2 MODE=ft LR=1e-5 bash examples/finetune.sh

# MeZO
MODEL=facebook/opt-1.3b TASK=SST2 MODE=ft LR=1e-7 BS=8 EPS=1e-3 bash examples/mezo.sh

# DPZero
MODEL=facebook/opt-1.3b TASK=SST2 MODE=ft LR=1e-7 BS=8 EPS=1e-3 DP_EPS=6.0 DP_CLIP=300 bash examples/dpzero.sh
```

More on hyperparameters can be found in our paper.
