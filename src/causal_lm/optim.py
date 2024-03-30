from torch import nn


def get_grouped_params(model: nn.Module, w_decay=0.1, no_decay=["bias", "LayerNorm.weight"]):
    params_with_wd, params_without_wd = [], []
    for n, p in model.named_parameters():
        if any(nd in n for nd in no_decay):
            params_without_wd.append(p)
        else:
            params_with_wd.append(p)
    return [
        {"params": params_with_wd, "weight_decay": w_decay},
        {"params": params_without_wd, "weight_decay": 0.0},
    ]
