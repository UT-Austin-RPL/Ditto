from src.third_party.ConvONets.config import get_model as ConvONets


def create_network(mode_opt):
    network = eval(mode_opt.network_type)(mode_opt)
    return network
