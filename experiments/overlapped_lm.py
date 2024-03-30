import pdb
from dataclasses import asdict, dataclass
from typing import Literal

import torch
from accelerate import Accelerator
from datasets import Dataset, DatasetDict, load_dataset
from tap import tapify
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, GPT2LMHeadModel, get_scheduler, set_seed

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
    output_dir: str = "output/overlapped_lm"
    comment: str = ""  # Add a comment to the run
    dbg: bool = False  # Start pdb, reduce the dataset size, and disable WandB

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
    )
    model = GPT2LMHeadModel(model_config)
    model_size = sum(t.numel() for t in model.parameters())
    print(f"GPT-2 size: {model_size/1000**2:.1f}M parameters")
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
    if args.dbg:
        pdb.set_trace()
    if not args.dbg:
        wandb.init(project="lm", config=asdict(args), notes=args.comment)
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    sampler = get_sampler_fn(args.train_sampler, tokenizer, args.context_length, key="text")
    eval_sampler = basic_sampling_fn(tokenizer, args.eval_context_length, key="text")
    lm_datasets = setup_data(sampler, eval_sampler, args.num_workers, args.dbg)
    train_loader = DataLoader(lm_datasets["train"], args.seed, batch_size=args.batch_size // args.gradient_accumulation_steps)
    val_loader = DataLoader(lm_datasets["valid"], args.seed, batch_size=args.eval_batch_size)
    test_loader = DataLoader(lm_datasets["test"], args.seed, batch_size=args.eval_batch_size)

    model = setup_model(tokenizer, args.context_length, args.n_layers)
    accelerator = Accelerator(mixed_precision="bf16")
    optimizer = AdamW(get_grouped_params(model), lr=5e-4)
    model, optimizer, train_loader, val_loader, test_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader, test_loader
    )

    num_update_steps_per_epoch = len(train_loader)
    num_training_steps = args.epochs * num_update_steps_per_epoch
    lr_scheduler = get_scheduler(
        name="linear",
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
            logits = model(batch["input_ids"]).logits
            loss = lm_loss(batch["input_ids"], logits)
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
