# Ditto: Building Digital Twins of Articulated Objects <br> through Interactive Perception

[Zhenyu Jiang](http://zhenyujiang.me), [Cheng-Chun Hsu](https://chengchunhsu.github.io/), [Yuke Zhu](https://www.cs.utexas.edu/~yukez/)


[Project](https://ut-austin-rpl.github.io/Ditto/) | arxiv

![intro](assets/pipeline.png)

## Introduction
<ins></ins>Ditto (<ins>Di</ins>gital <ins>T</ins>wins of Ar<ins>t</ins>iculated <ins>O</ins>bjects) is a model that reconstructs part-level geometry and articulation model of an articulated object given observations before and after an interaction. Specifically, we use a PointNet++ encoder to encoder the input point cloud observations, and fuse the subsampled point features with a simple attention layer. Then we use two independent decoders to propagate the fused point features into two sets of dense point features, for geometry reconstruction and articulation estimation separately. We construct feature grid/planes by projecting and pooling the point features, and query local features from the constructed feature grid/planes. Conditioning on local features, we use different decoders to predict occupancy, segmentation and joint parameters with respect to the query points. At then end, we can extract explicit geometry and articulation model from the implicit decoders.

If you find our work useful in your research, please consider [citing](assets/ditto.bib).

## Installation

1. Create a conda environment and install required packages.

```bash
conda env create -f conda_env_gpu.yaml -n Ditto
```

You can change the `pytorch` and `cuda` version in conda_env_gpu.yaml.

2. Build ConvONets dependents by running `python scripts/convonet_setup.py build_ext --inplace`.

3. Download the [data](#data-and-pre-trained-models), then unzip the `data.zip` under the repo's root.

## Training

```Python
# single GPU
python run.py experiment=Ditto_s2m

# multiple GPUs
python run.py trainer.gpus=4 +trainer.accelerator='ddp' experiment=Ditto_s2m

# multiple GPUs + wandb logging
python run.py trainer.gpus=4 +trainer.accelerator='ddp' logger=wandb logger.wandb.group=s2m experiment=Ditto_s2m
```

## Testing

```Python
# only support single GPU
python run_test.py experiment=Ditto_s2m trainer.resume_from_checkpoint=/path/to/trained/model/
```

## Data and pre-trained models

Data: [here](https://utexas.box.com/s/1wiynn7ql42c3mi1un7ynncfxr86ep22). Remeber to cite [Shape2Motion](assets/s2m.bib) and [Abbatematteo *et al.*](assets/syn.bib) as well as [Ditto](assets/ditto.bib) when using these datasets.

Pre-trained models: [Shape2Motion dataset](https://utexas.box.com/s/ktckf75xo33plf5nidyvqz9bn20jwv06), [Synthetic dataset](https://utexas.box.com/s/zbf5bja20n2w6umryb1bcfbbcm3h2ysn).

## Useful tips

1. Run `eval "$(python run.py -sc install=bash)"` under the root directory, you can have auto-completion for commandline options.

2. Install pre-commit hooks by `pip install pre-commit; pre-commit install`, then you can have automatic formatting before each commit.

## Related Repositories

1. Our code is based on this fantastic template[Lightning-Hydra-Template](https://github.com/ashleve/lightning-hydra-template)

2. We use [ConvONets](https://github.com/autonomousvision/convolutional_occupancy_networks) as our backbone.
