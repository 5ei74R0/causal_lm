import pdb
from dataclasses import asdict, dataclass
from typing import Literal, Optional, Tuple, Union

import torch
from accelerate import Accelerator
from datasets import Dataset, DatasetDict, load_dataset
from tap import tapify
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, GPT2LMHeadModel, GPT2Model, get_scheduler, set_seed
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

import wandb
from causal_lm.data import ReproducibleDataLoader as DataLoader
from causal_lm.data import basic_sampling_fn, overlapped_sampling_fn
from causal_lm.loss import lm_loss
from causal_lm.optim import get_grouped_params


@dataclass
class Args:
    # general
    seed: int = 5740
    num_workers: int = 8
    output_dir: str = "output/start_from_bos_lm"
    comment: str = ""  # Add a comment to the run
    dbg: bool = False  # Reduce the dataset size, and disable WandB
    pdb: bool = False  # Start pdb, reduce the dataset size, and disable WandB

    # training
    train_sampler: Literal["basic", "overlapped"] = "basic"
    batch_size: int = 128  # must be divisible by gradient_accumulation_steps
    context_length: int = 128
    epochs: int = 2
    gradient_accumulation_steps: int = 4
    lr: float = 5e-4

    # eval
    eval_steps: int = 500
    eval_batch_size: int = 8
    eval_context_length: int = 256

    # model customization
    n_layers: int = 12


def get_sampler_fn(sampler_type: str, tokenizer, context_length, key="text"):
    if sampler_type == "basic":
        return basic_sampling_fn(tokenizer, context_length, key=key)
    elif sampler_type == "overlapped":
        return overlapped_sampling_fn(tokenizer, context_length, key=key)
    raise ValueError(f"Unknown sampler type: {sampler_type}")


def setup_data(sampler, eval_sampler, num_workers, dbg_mode) -> DatasetDict:
    suffix = "[:1%]" if dbg_mode else ""
    train_set: Dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train" + suffix)
    val_set: Dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
    test_set: Dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")

    train_set = train_set.map(sampler, num_proc=num_workers, batched=True, remove_columns=train_set.column_names)
    val_set = val_set.map(eval_sampler, num_proc=num_workers, batched=True, remove_columns=val_set.column_names)
    test_set = test_set.map(eval_sampler, num_proc=num_workers, batched=True, remove_columns=test_set.column_names)

    dataset_dict = DatasetDict(train=train_set, valid=val_set, test=test_set)
    dataset_dict.set_format(type="torch")  # provide batches as PyTorch tensors
    return dataset_dict


def setup_model(tokenizer, context_length, n_layers) -> GPT2LMHeadModel:
    model_config = AutoConfig.from_pretrained(
        "gpt2",
        vocab_size=len(tokenizer),
        n_ctx=context_length,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        # additional customization
        n_layer=n_layers,
        n_head=n_layers,
        n_embd=n_layers * 64,
        output_hidden_states=True,  # to calculate the interpolation loss
    )
    model = GPT2LMHeadModel(model_config)
    model_size = sum(t.numel() for t in model.parameters())
    print(f"GPT-2 size: {model_size/1000**2:.1f}M parameters")

    class gpt2_custom_forward_fn:
        def __init__(self, gpt2_ref, tokenizer) -> None:
            self.ref: GPT2Model = gpt2_ref
            self.bos_token_id = tokenizer.bos_token_id

        def __call__(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
        ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
            self.noisy_wte.to(self.ref.device)
            output_attentions = (
                output_attentions if output_attentions is not None else self.ref.config.output_attentions
            )
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.ref.config.output_hidden_states
            )
            use_cache = use_cache if use_cache is not None else self.ref.config.use_cache
            return_dict = return_dict if return_dict is not None else self.ref.config.use_return_dict

            if input_ids is not None and inputs_embeds is not None:
                raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
            elif input_ids is not None:
                self.ref.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
                input_shape = input_ids.size()
                input_ids = input_ids.view(-1, input_shape[-1])
                batch_size = input_ids.shape[0]
            elif inputs_embeds is not None:
                input_shape = inputs_embeds.size()[:-1]
                batch_size = inputs_embeds.shape[0]
            else:
                raise ValueError("You have to specify either input_ids or inputs_embeds")

            device = input_ids.device if input_ids is not None else inputs_embeds.device

            if token_type_ids is not None:
                token_type_ids = token_type_ids.view(-1, input_shape[-1])

            if past_key_values is None:
                past_length = 0
                past_key_values = tuple([None] * len(self.ref.h))
            else:
                past_length = past_key_values[0][0].size(-2)
            if position_ids is None:
                position_ids = torch.arange(
                    past_length, input_shape[-1] + past_length, dtype=torch.long, device=device
                )
                position_ids = position_ids.unsqueeze(0)

            # GPT2Attention mask.
            if attention_mask is not None:
                if batch_size <= 0:
                    raise ValueError("batch_size has to be defined and > 0")
                attention_mask = attention_mask.view(batch_size, -1)
                # We create a 3D attention mask from a 2D tensor mask.
                # Sizes are [batch_size, 1, 1, to_seq_length]
                # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
                # this attention mask is more simple than the triangular masking of causal attention
                # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
                attention_mask = attention_mask[:, None, None, :]

                # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
                # masked positions, this operation will create a tensor which is 0.0 for
                # positions we want to attend and the dtype's smallest value for masked positions.
                # Since we are adding it to the raw scores before the softmax, this is
                # effectively the same as removing these entirely.
                attention_mask = attention_mask.to(dtype=self.ref.dtype)  # fp16 compatibility
                attention_mask = (1.0 - attention_mask) * torch.finfo(self.ref.dtype).min

            # If a 2D or 3D attention mask is provided for the cross-attention
            # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.ref.config.add_cross_attention and encoder_hidden_states is not None:
                encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
                encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
                if encoder_attention_mask is None:
                    encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
                encoder_attention_mask = self.ref.invert_attention_mask(encoder_attention_mask)
            else:
                encoder_attention_mask = None

            # Prepare head mask if needed
            # 1.0 in head_mask indicate we keep the head
            # attention_probs has shape bsz x n_heads x N x N
            # head_mask has shape n_layer x batch x n_heads x N x N
            head_mask = self.ref.get_head_mask(head_mask, self.ref.config.n_layer)

            if inputs_embeds is None:
                inputs_embeds = self.ref.wte(input_ids)
                bos_embeds = self.ref.wte(torch.full_like(input_ids, self.bos_token_id, device=device))
            position_embeds = self.ref.wpe(position_ids)
            hidden_states = inputs_embeds + position_embeds
            start_states = bos_embeds + position_embeds

            if token_type_ids is not None:
                token_type_embeds = self.ref.wte(token_type_ids)
                hidden_states = hidden_states + token_type_embeds

            hidden_states = self.ref.drop(hidden_states)

            output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

            if self.ref.gradient_checkpointing and self.ref.training:
                if use_cache:
                    print(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )  # logger.warning_once
                    use_cache = False

            presents = () if use_cache else None
            all_self_attentions = () if output_attentions else None
            all_cross_attentions = () if output_attentions and self.ref.config.add_cross_attention else None
            all_hidden_states = () if output_hidden_states else None
            for i, (block, layer_past) in enumerate(zip(self.ref.h, past_key_values)):
                # Model parallel
                if self.ref.model_parallel:
                    torch.cuda.set_device(hidden_states.device)
                    # Ensure layer_past is on same device as hidden_states (might not be correct)
                    if layer_past is not None:
                        layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
                    # Ensure that attention_mask is always on the same device as hidden_states
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(hidden_states.device)
                    if isinstance(head_mask, torch.Tensor):
                        head_mask = head_mask.to(hidden_states.device)
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                if self.ref.gradient_checkpointing and self.ref.training:
                    outputs = self.ref._gradient_checkpointing_func(
                        block.__call__,
                        hidden_states,
                        None,
                        attention_mask,
                        head_mask[i],
                        encoder_hidden_states,
                        encoder_attention_mask,
                        use_cache,
                        output_attentions,
                        start_states if i == 0 else None,  # input bos embeddings for the first layer
                    )
                else:
                    outputs = block(
                        hidden_states,
                        layer_past=layer_past,
                        attention_mask=attention_mask,
                        head_mask=head_mask[i],
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=encoder_attention_mask,
                        use_cache=use_cache,
                        output_attentions=output_attentions,
                        res_input=start_states if i == 0 else None,  # input bos embeddings for the first layer
                    )

                hidden_states = outputs[0]
                if use_cache is True:
                    presents = presents + (outputs[1],)

                if output_attentions:
                    all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                    if self.ref.config.add_cross_attention:
                        all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

                # Model Parallel: If it's the last layer for that device, put things on the next device
                if self.ref.model_parallel:
                    for k, v in self.ref.device_map.items():
                        if i == v[-1] and "cuda:" + str(k) != self.ref.last_device:
                            hidden_states = hidden_states.to("cuda:" + str(k + 1))

            hidden_states = self.ref.ln_f(hidden_states)

            hidden_states = hidden_states.view(output_shape)
            # Add last hidden state
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if not return_dict:
                return tuple(
                    v
                    for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
                    if v is not None
                )

            return BaseModelOutputWithPastAndCrossAttentions(
                last_hidden_state=hidden_states,
                past_key_values=presents,
                hidden_states=all_hidden_states,
                attentions=all_self_attentions,
                cross_attentions=all_cross_attentions,
            )

    class layer_forward_fn:
        def __init__(self, model_ref) -> None:
            self.ref = model_ref

        def __call__(
            self,
            hidden_states: Optional[Tuple[torch.FloatTensor]],
            layer_past: Optional[Tuple[torch.Tensor]] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = False,
            output_attentions: Optional[bool] = False,
            res_input: Optional[torch.FloatTensor] = None,
        ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
            # STARTING_DISTRIBUTION = torch.rand_like(hidden_states).to(hidden_states.device) * hidden_states.abs().mean() * math.sqrt(hidden_states.size(-1))
            residual = hidden_states
            hidden_states = self.ref.ln_1(hidden_states)
            attn_outputs = self.ref.attn(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
            outputs = attn_outputs[1:]
            # residual connection
            hidden_states = attn_output + (residual if res_input is None else res_input)

            if encoder_hidden_states is not None:
                # add one self-attention block for cross-attention
                if not hasattr(self.ref, "crossattention"):
                    raise ValueError(
                        f"If `encoder_hidden_states` are passed, {self.ref} has to be instantiated with "
                        "cross-attention layers by setting `config.add_cross_attention=True`"
                    )
                residual = hidden_states
                hidden_states = self.ref.ln_cross_attn(hidden_states)
                cross_attn_outputs = self.ref.crossattention(
                    hidden_states,
                    attention_mask=attention_mask,
                    head_mask=head_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    output_attentions=output_attentions,
                )
                attn_output = cross_attn_outputs[0]
                # residual connection
                hidden_states = residual + attn_output
                outputs = outputs + cross_attn_outputs[2:]  # add cross attentions if we output attention weights

            residual = hidden_states
            hidden_states = self.ref.ln_2(hidden_states)
            feed_forward_hidden_states = self.ref.mlp(hidden_states)
            # residual connection
            hidden_states = residual + feed_forward_hidden_states

            if use_cache:
                outputs = (hidden_states,) + outputs
            else:
                outputs = (hidden_states,) + outputs[1:]

            return outputs  # hidden_states, present, (attentions, cross_attentions)

    model.transformer.forward = gpt2_custom_forward_fn(model.transformer)
    for attn_ffn in model.transformer.h:
        attn_ffn.forward = layer_forward_fn(attn_ffn)
    return model


def register_metrics(dbg=False):
    if not dbg:
        wandb.define_metric("update-steps")
        # define which metrics will be plotted against it
        wandb.define_metric("loss-eval", step_metric="update-steps")
        wandb.define_metric("ppl-eval", step_metric="update-steps")
        wandb.define_metric("loss-train", step_metric="update-steps")
        wandb.define_metric("lr", step_metric="update-steps")


def logging(eval_loss, perplexity, loss, lr_scheduler, completed_steps, dbg=False):
    metrics = {
        "loss-eval": eval_loss,
        "ppl-eval": perplexity,
        "loss-train": loss.mean().item(),
        "lr": lr_scheduler.get_lr()[0],
        "update-steps": completed_steps,
    }
    if dbg:
        print(metrics)
    else:
        wandb.log(metrics)


def run():
    args = tapify(Args)
    if args.pdb:
        pdb.set_trace()
        args.dbg = True
    if not args.dbg:
        wandb.init(project="lm", config=asdict(args), notes=args.comment, mode="offline")
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    sampler = get_sampler_fn(args.train_sampler, tokenizer, args.context_length, key="text")
    eval_sampler = basic_sampling_fn(tokenizer, args.eval_context_length, key="text")
    lm_datasets = setup_data(sampler, eval_sampler, args.num_workers, args.dbg)
    train_loader = DataLoader(
        lm_datasets["train"], args.seed, batch_size=args.batch_size // args.gradient_accumulation_steps
    )
    val_loader = DataLoader(lm_datasets["valid"], args.seed, batch_size=args.eval_batch_size)
    test_loader = DataLoader(lm_datasets["test"], args.seed, batch_size=args.eval_batch_size)

    model = setup_model(tokenizer, args.context_length, args.n_layers)
    accelerator = Accelerator(mixed_precision="bf16")
    optimizer = AdamW(get_grouped_params(model), lr=5e-4)
    model, optimizer, train_loader, val_loader, test_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader, test_loader
    )

    num_update_steps_per_epoch = len(train_loader)
    num_training_steps = args.epochs * num_update_steps_per_epoch // args.gradient_accumulation_steps
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=1_000,
        num_training_steps=num_training_steps,
    )

    @torch.inference_mode()
    def evaluate(loader: DataLoader):
        model.eval()
        losses = []
        for step, batch in enumerate(loader):
            outputs = model(batch["input_ids"], labels=batch["input_ids"])
            losses.append(accelerator.gather(outputs.loss).unsqueeze(0))
        loss = torch.mean(torch.cat(losses))
        try:
            perplexity = torch.exp(loss)
        except OverflowError:
            perplexity = float("inf")
        return loss.item(), perplexity.item()

    eval_loss, perplexity = evaluate(val_loader)
    if not args.dbg:
        wandb.log({"loss-eval": eval_loss, "ppl-eval": perplexity})
        wandb.watch(model, log="gradients", log_graph=True)

    model.train()
    completed_steps = 0
    baseline_ppl = []
    for epoch in range(args.epochs):
        for step, batch in tqdm(enumerate(train_loader, start=1), total=num_training_steps):
            output_ = model(batch["input_ids"])
            logits = output_.logits
            loss = lm_loss(batch["input_ids"], logits)
            # loss = interpolation_lm_loss(batch["input_ids"], logits, noisy_wte, output_.hidden_states)
            if step % 100 == 0:
                accelerator.print(
                    {
                        "lr": lr_scheduler.get_lr(),
                        "samples": completed_steps * args.batch_size,
                        "steps": completed_steps,
                        "loss/train": loss.mean().item() * args.gradient_accumulation_steps,
                    }
                )
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss.sum())
            if step % args.gradient_accumulation_steps == 0:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                completed_steps += 1
            if (step % (args.eval_steps * args.gradient_accumulation_steps)) == 0:
                eval_loss, perplexity = evaluate(val_loader)
                accelerator.print({"loss/eval": eval_loss, "perplexity": perplexity})
                baseline_ppl.append(perplexity)
                logging(eval_loss, perplexity, loss, lr_scheduler, completed_steps, args.dbg)
                model.train()
                accelerator.wait_for_everyone()
                # unwrapped_model = accelerator.unwrap_model(model)
                # unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)
                # if accelerator.is_main_process:
                #     tokenizer.save_pretrained(output_dir)
                #     repo.push_to_hub(
                #         commit_message=f"Training in progress step {step}", blocking=False
                #     )
    eval_loss, perplexity = evaluate(test_loader)
    if not args.dbg:
        wandb.log({"ppl-final": perplexity, "loss-test": eval_loss})
    print(f"Final perplexity: {perplexity:.2f}")

    if not args.dbg and wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    run()
