import torch


@torch.compile(mode="reduce-overhead")
def lambda_returns(
    *,
    rewards: torch.Tensor,
    terminations: torch.Tensor,
    next_q: torch.Tensor,
    gamma: float,
    return_lambda: float,
) -> torch.Tensor:
    T = rewards.size(0)
    returns = torch.zeros_like(rewards)
    discount = gamma * (1.0 - terminations)
    q_weighted = (1.0 - return_lambda) * next_q
    _return = rewards[-1] + discount[-1] * next_q[-1]
    returns[-1] = _return
    for t in range(T - 2, -1, -1):
        _return = rewards[t] + discount[t] * (q_weighted[t] + return_lambda * _return)
        returns[t] = _return
    return returns
