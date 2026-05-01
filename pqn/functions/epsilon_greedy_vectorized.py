import torch
import numpy


def epsilon_greedy_vectorized(
    q_values: torch.Tensor, eps: torch.Tensor
) -> numpy.ndarray:
    eps = eps.to(q_values.device)
    if eps.ndim == 1 and eps.shape[0] != q_values.shape[0]:
        eps = eps[0]

    num_envs, action_dim = q_values.shape
    greedy_actions = torch.argmax(q_values, dim=-1)
    random_actions = torch.randint(0, action_dim, (num_envs,), device=q_values.device)

    mask = torch.rand(num_envs, device=q_values.device) < eps
    final_actions = torch.where(mask, random_actions, greedy_actions)
    return final_actions.cpu().numpy()
