# SISO_RL

(let's add wandb tracking, very nice to have in an RL repo)

## Install 

Installation instructions.

```bash
conda create -n siso python=3.12
conda activate siso
pip install -r requirements.txt
```

## Run

How to train an RL agent considering a single input single output system.

```bash
python train_siso.py
```

How to train an RL agent considering a multiple input multiple output system (two inputs and two outputs for this example)..

```bash
python train_mimo.py
```


