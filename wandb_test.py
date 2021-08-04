# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import wandb


# %%
import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

import torch
import torchvision
import torchvision.datasets as datasets


# %%
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)


# %%
WANDB_NOTEBOOK_NAME = os.environ['WANDB_NOTEBOOK_NAME']


# %%
# 1. Start a new run
wandb.init(project='mnist-noise-autoencoder', entity='jorgecanedo')


# %%
# 2. Save model inputs and hyperparameters
config = wandb.config
config.learning_rate = 0.01


# %%
mnist_trainset = datasets.MNIST(
    root='./data', train=True, download=True, transform=None)
df = pd.DataFrame(mnist_trainset)


# %%
print(df.head())


# %%
# # 3. Log gradients and model parameters
# wandb.watch(model)
# for batch_idx, (data, target) in enumerate(train_loader):
#   ...
#   if batch_idx % args.log_interval == 0:
#     # 4. Log metrics to visualize performance
#     wandb.log({"loss": loss})
