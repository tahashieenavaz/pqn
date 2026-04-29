import torch


class PQN:
    def __init__(self, λ: float = 0.65, γ: float = 0.99):
        params = locals()
        params.pop("self")

        self.__initialize_hyper_parameters(params)

    def __initialize_hyper_parameters(self, params: dict):
        for key, value in params:
            setattr(self, key, value)
