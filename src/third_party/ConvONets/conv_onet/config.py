import omegaconf
from torch import nn

from src.third_party.ConvONets.conv_onet import generation_two_stage as generation
from src.third_party.ConvONets.conv_onet import models
from src.third_party.ConvONets.conv_onet.models import (
    ConvolutionalOccupancyNetworkGeoArt,
    ConvolutionalOccupancyNetworkGeometry,
)
from src.third_party.ConvONets.encoder import encoder_dict


def get_model(cfg, dataset=None, **kwargs):
    """Return the Occupancy Network model.

    Args:
        cfg (dict): imported yaml config
        dataset (dataset): dataset
    """
    convonet_type = cfg["convonet_type"]
    decoder = cfg["decoder"]
    encoder = cfg["encoder"]
    c_dim = cfg["c_dim"]
    decoder_kwargs = cfg["decoder_kwargs"]
    encoder_kwargs = cfg["encoder_kwargs"]
    padding = cfg["padding"]
    if padding is None:
        padding = 0.1

    # for pointcloud_crop
    try:
        encoder_kwargs["unit_size"] = cfg["data"]["unit_size"]
        decoder_kwargs["unit_size"] = cfg["data"]["unit_size"]
    except:
        pass
    # local positional encoding
    if "local_coord" in cfg.keys():
        encoder_kwargs["local_coord"] = cfg["local_coord"]
        decoder_kwargs["local_coord"] = cfg["local_coord"]
    if "pos_encoding" in cfg:
        encoder_kwargs["pos_encoding"] = cfg["pos_encoding"]
        decoder_kwargs["pos_encoding"] = cfg["pos_encoding"]

    decoders = []
    if isinstance(cfg["decoder"], list) or isinstance(
        cfg["decoder"], omegaconf.listconfig.ListConfig
    ):
        for i, d_name in enumerate(cfg["decoder"]):
            decoder = models.decoder_dict[d_name](padding=padding, **decoder_kwargs[i])
            decoders.append(decoder)
    else:
        decoder = models.decoder_dict[cfg["decoder"]](padding=padding, **decoder_kwargs)
        decoders.append(decoder)

    if encoder == "idx":
        encoder = nn.Embedding(len(dataset), c_dim)
    elif encoder is not None:
        encoder = encoder_dict[encoder](c_dim=c_dim, padding=padding, **encoder_kwargs)
    else:
        encoder = None

    if len(decoders) == 1:
        model = eval(convonet_type)(decoder, encoder)
    else:
        model = eval(convonet_type)(decoders, encoder)

    return model


def get_generator(model, cfg, **kwargs):
    """Returns the generator object.

    Args:
        model (nn.Module): Occupancy Network model
        cfg (dict): imported yaml config
    """
    vol_bound = None
    vol_info = None

    generator = generation.Generator3D(
        model,
        threshold=cfg["test"]["threshold"],
        resolution0=cfg["generation"]["resolution_0"],
        upsampling_steps=cfg["generation"]["upsampling_steps"],
        sample=cfg["generation"]["use_sampling"],
        refinement_step=cfg["generation"]["refinement_step"],
        simplify_nfaces=cfg["generation"]["simplify_nfaces"],
        input_type=cfg["data"]["input_type"],
        padding=cfg["data"]["padding"],
        vol_info=vol_info,
        vol_bound=vol_bound,
    )
    return generator
