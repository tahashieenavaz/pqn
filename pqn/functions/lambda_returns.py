import torch


@torch.jit.script
def lambda_returns(
    rewards: torch.Tensor,
    terminations: torch.Tensor,
    next_q: torch.Tensor,
    gamma: float,
    return_lambda: float,
) -> torch.Tensor:
    T = rewards.size(0)
    out = torch.zeros_like(rewards)
    ret = rewards[-1] + gamma * next_q[-1] * (1 - terminations[-1])
    out[-1] = ret

    for t in range(T - 2, -1, -1):
        bootstrap = next_q[t]
        td_target = rewards[t] + gamma * (1 - terminations[t]) * bootstrap
        ret = td_target + gamma * return_lambda * (1 - terminations[t]) * (
            ret - bootstrap
        )
        out[t] = ret
    return out
