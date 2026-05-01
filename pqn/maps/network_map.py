from ..networks import QNetwork
from ..networks import DuellingNetwork

network_map = {"q": QNetwork, "duelling": DuellingNetwork}
