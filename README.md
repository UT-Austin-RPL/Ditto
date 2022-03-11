# Ditto: Building Digital Twins of Articulated Objects from Interaction

[Zhenyu Jiang](http://zhenyujiang.me), [Cheng-Chun Hsu](https://chengchunhsu.github.io/), [Yuke Zhu](https://www.cs.utexas.edu/~yukez/)

CVPR 2022

[Project](https://ut-austin-rpl.github.io/Ditto/) | [arxiv](https://arxiv.org/abs/2202.08227)

![intro](assets/pipeline.png)

## Introduction
<ins></ins>Ditto (<ins>Di</ins>gital <ins>T</ins>wins of Ar<ins>t</ins>iculated <ins>O</ins>bjects) is a model that reconstructs part-level geometry and articulation model of an articulated object given observations before and after an interaction. Specifically, we use a PointNet++ encoder to encoder the input point cloud observations, and fuse the subsampled point features with a simple attention layer. Then we use two independent decoders to propagate the fused point features into two sets of dense point features, for geometry reconstruction and articulation estimation separately. We construct feature grid/planes by projecting and pooling the point features, and query local features from the constructed feature grid/planes. Conditioning on local features, we use different decoders to predict occupancy, segmentation and joint parameters with respect to the query points. At then end, we can extract explicit geometry and articulation model from the implicit decoders.

If you find our work useful in your research, please consider [citing](#citing).

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

## Demo

[Here](notebooks/demo_depth_map.ipynb) is a minimum demo that starts from multiview depth maps before and after interaction and ends with a reconstructed digital twin. To run the demo, you need to install this [library](https://github.com/Steve-Tod/utils3d) for visualization, and use [this model weight](https://utexas.box.com/s/ktckf75xo33plf5nidyvqz9bn20jwv06).

We provide the posed depth images of a real word muisc box to run the demo. You can download from [here](https://utexas.box.com/s/ujb2ky8y9vaog7nheth1n3tmm1rgx9t7) and put it under `data`. You can also run demo your own data that follows the same format.

## Data and pre-trained models

Data: [here](https://utexas.box.com/s/1wiynn7ql42c3mi1un7ynncfxr86ep22). Remeber to cite [Shape2Motion](assets/s2m.bib) and [Abbatematteo *et al.*](#citing) as well as [Ditto](assets/ditto.bib) when using these datasets.

Pre-trained models: [Shape2Motion dataset](https://utexas.box.com/s/a4h001b3ciicrt3f71t4xd3wjsm04be7), [Synthetic dataset](https://utexas.box.com/s/zbf5bja20n2w6umryb1bcfbbcm3h2ysn).

## Useful tips

1. Run `eval "$(python run.py -sc install=bash)"` under the root directory, you can have auto-completion for commandline options.

2. Install pre-commit hooks by `pip install pre-commit; pre-commit install`, then you can have automatic formatting before each commit.

## Related Repositories

1. Our code is based on this fantastic template [Lightning-Hydra-Template](https://github.com/ashleve/lightning-hydra-template).

2. We use [ConvONets](https://github.com/autonomousvision/convolutional_occupancy_networks) as our backbone.

## Citing
```
@inproceedings{jiang2022ditto,
   title={Ditto: Building Digital Twins of Articulated Objects from Interaction},
   author={Jiang, Zhenyu and Hsu, Cheng-Chun and Zhu, Yuke},
   booktitle={arXiv preprint arXiv:2202.08227},
   year={2022}
}
```
