# DSMT-T5

Code for paper "Dual-Stage Multi-Task Syntax-Oriented Pre-Training for Syntactically Controlled Paraphrase Generation"

## Prerequistes:

Install python3.8

Install the following packages:


  - torch==1.13.1
  - transformers
  - accelerate==0.24.0
  - jsonargparse

## DSMT Pre-training

change your directory to `curriculum_pre_training`

run

```
python training_watchdog.py \
    --config /path/to/config/file
    --output_dir /path/to/output/dir
```

## Fine-Tuning on SCPG

run

```
python main.py para_train \
    --config /path/to/config/file
    --output_dir /path/to/output/dir
```
