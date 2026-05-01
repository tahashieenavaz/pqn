import torch

optimizer_map = {
    "radam": torch.optim.RAdam,
    "nadam": torch.optim.NAdam,
    "adamw": torch.optim.AdamW,
    "adam": torch.optim.Adam,
}
