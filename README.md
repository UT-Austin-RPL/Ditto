# Ditto

## Installation

```bash
conda env create -f conda_env_gpu.yaml -n Ditto
# install torch-scatter
pip install https://data.pyg.org/whl/torch-1.8.0%2Bcu102/torch_scatter-2.0.8-cp38-cp38-linux_x86_64.whl
python scripts/convonet_setup.py build_ext --inplace
```

## Train

```Python
# single GPU
python run.py experiment=syn_cross_cat4_ppp_attn_v3_3d_2d

# multiple GPUs
python run.py trainer.gpus=4 +trainer.accelerator='ddp' experiment=syn_cross_cat4_ppp_attn_v3_3d_2d

# multiple GPUs + wandb logging
python run.py trainer.gpus=4 +trainer.accelerator='ddp' logger=wandb logger.wandb.group=synthetic experiment=syn_cross_cat4_ppp_attn_v3_3d_2d
```
