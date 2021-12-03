import torch
import torch.nn as nn
from torch import distributions as dist

from src.third_party.ConvONets.conv_onet.models import decoder

# Decoder dictionary
decoder_dict = {
    "simple_fc": decoder.FCDecoder,
    "simple_local": decoder.LocalDecoder,
    "simple_local_v1": decoder.LocalDecoderV1,
}


class ConvolutionalOccupancyNetworkGeoArt(nn.Module):
    def __init__(self, decoders, encoder=None):
        super().__init__()

        (
            self.decoder_occ,
            self.decoder_seg,
            self.decoder_joint_type,
            self.decoder_revolute,
            self.decoder_prismatic,
        ) = decoders

        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = None

    def forward(self, input_0, input_1, p_occ, p_seg, return_feature=False, **kwargs):
        """Performs a forward pass through the network.
        Args:
            p (tensor): sampled points, B*N*C
            inputs (tensor): conditioning input, B*N*3
        """
        #############
        c = self.encoder(input_0, input_1)
        occ_logits = self.decoder_occ(p_occ, c, **kwargs)
        seg_logits = self.decoder_seg(p_seg, c, **kwargs)
        joint_type_logits = self.decoder_joint_type(p_seg, c, **kwargs)
        joint_param_r = self.decoder_revolute(p_seg, c, **kwargs)
        joint_param_p = self.decoder_prismatic(p_seg, c, **kwargs)

        if return_feature:
            return (
                occ_logits,
                seg_logits,
                joint_type_logits,
                joint_param_r,
                joint_param_p,
                c,
            )
        else:
            return (
                occ_logits,
                seg_logits,
                joint_type_logits,
                joint_param_r,
                joint_param_p,
            )

    def decode_joints(self, p, c):
        """Returns occupancy probabilities for the sampled points.
        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        """
        joint_type_logits = self.decoder_joint_type(p, c)
        joint_param_r = self.decoder_revolute(p, c)
        joint_param_p = self.decoder_prismatic(p, c)
        return joint_type_logits, joint_param_r, joint_param_p

    def decode_occ(self, p, c, **kwargs):
        """Returns occupancy probabilities for the sampled points.
        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        """

        logits = self.decoder_occ(p, c, **kwargs)
        return logits

    def decode_seg(self, p, c, **kwargs):
        """Returns occupancy probabilities for the sampled points.
        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        """

        logits = self.decoder_seg(p, c, **kwargs)
        return logits

    def encode_inputs(self, input_0, input_1):
        """Encodes the input.
        Args:
            input (tensor): the input
        """

        if self.encoder is not None:
            c = self.encoder(input_0, input_1)
        else:
            # Return inputs?
            c = torch.empty(input_0.size(0), 0)

        return c


class ConvolutionalOccupancyNetworkGeometry(nn.Module):
    def __init__(self, decoder, encoder=None, device=None):
        super().__init__()

        self.decoder = decoder.to(device)

        if encoder is not None:
            self.encoder = encoder.to(device)
        else:
            self.encoder = None

        self._device = device

    def forward(self, inputs, p_occ, **kwargs):
        """Performs a forward pass through the network.
        Args:
            inputs (tensor): conditioning input, B*N*3
            sample (bool): whether to sample for z
            p_occ (tensor): occ query points, B*N_P*3
        """
        #############
        c = self.encode_inputs(inputs)
        logits_occ = self.decoder(p_occ, c, **kwargs)
        return logits_occ

    def encode_inputs(self, inputs):
        """Encodes the input.
        Args:
            input (tensor): the input
        """

        if self.encoder is not None:
            c = self.encoder(inputs)
        else:
            # Return inputs?
            c = torch.empty(inputs.size(0), 0)

        return c

    def decode_occ(self, p, c, **kwargs):
        """Returns occupancy probabilities for the sampled points.
        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        """

        logits = self.decoder(p, c, **kwargs)
        p_r = dist.Bernoulli(logits=logits)
        return p_r
