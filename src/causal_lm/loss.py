from torch import Tensor as Tn
from torch.nn import CrossEntropyLoss, MSELoss


def lm_loss(inputs: Tn, logits: Tn) -> Tn:
    # Shift so that tokens < n predict n
    shift_labels = inputs[..., 1:].contiguous()
    shift_logits = logits[..., :-1, :].contiguous()
    # Calculate per-token loss
    loss_fn = CrossEntropyLoss(reduction="none")
    loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    # Resize and average loss per sample
    loss_per_sample = loss.view(shift_logits.size(0), shift_logits.size(1)).mean(axis=1)
    return loss_per_sample


def _interpolate(i_logits: Tn, o_logits: Tn, step: int) -> list[Tn]:
    interpolated = [i_logits]
    for i in range(1, step + 1):
        interpolated.append(i_logits + (o_logits - i_logits) * i / step)
    return interpolated


def _interpolate_exp_decay(i_logits: Tn, o_logits: Tn, step: int) -> list[Tn]:
    interpolated = [i_logits]
    prev = i_logits
    direction = o_logits - i_logits
    length = 2
    for _ in range(1, step):
        interpolated.append(prev + direction / length)
        prev = interpolated[-1]
        length *= 2
    interpolated.append(o_logits)
    return interpolated


def interpolation_lm_loss(inputs: Tn, logits: Tn, embed_fn, hidden_states: Tn, exp_decay: bool = False) -> Tn:
    # Shift so that tokens < n predict n
    shift_labels = inputs[..., 1:].contiguous()
    shift_logits = logits[..., :-1, :].contiguous()
    # Calculate per-token loss
    loss_fn = CrossEntropyLoss(reduction="none")
    mse_fn = MSELoss(reduction="none")
    loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    # Calculate noise-to-logits interpolation loss
    in_logits = embed_fn(inputs[..., :-1].contiguous())
    out_logits = embed_fn(shift_labels)
    interpolate_fn = _interpolate if not exp_decay else _interpolate_exp_decay
    interpolated_logits = interpolate_fn(in_logits, out_logits, len(hidden_states))
    for _hidden, _target in zip(hidden_states[1:], interpolated_logits[1:]):
        shift_hidden = _hidden[..., :-1, :].contiguous()
        mse_val = mse_fn(shift_hidden.view(-1, shift_hidden.size(-1)), _target.view(-1, _target.size(-1)))
        loss += mse_val.mean(axis=-1)
    return loss


def denoising_lm_loss(inputs, logits, hidden_states):
    # Shift so that tokens < n predict n
    shift_labels = inputs[..., 1:].contiguous()
    shift_logits = logits[..., :-1, :].contiguous()
    # Calculate per-token loss
    loss_fn = CrossEntropyLoss(reduction="none")
    loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    # Calculate noise-to-logits interpolation loss
    raise NotImplementedError("Denoising loss not implemented yet")
