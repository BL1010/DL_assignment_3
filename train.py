"""
train.py — Training Pipeline, Inference & Evaluation
DA6401 Assignment 3: "Attention Is All You Need"
"""

import argparse
import os
import random
import numpy as np

import torch
import torch.nn as nn
import wandb

from tqdm import tqdm
from torch.utils.data import DataLoader

from nltk.translate.bleu_score import corpus_bleu

from dataset import (
    Multi30kDataset,
    CollateFn,
)

from model import (
    Transformer,
    make_src_mask,
    make_tgt_mask,
)

from lr_scheduler import NoamScheduler


# ══════════════════════════════════════════════════════════════════════
# SEED
# ══════════════════════════════════════════════════════════════════════

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ══════════════════════════════════════════════════════════════════════
# LABEL SMOOTHING LOSS
# ══════════════════════════════════════════════════════════════════════

class LabelSmoothingLoss(nn.Module):

    def __init__(
        self,
        vocab_size,
        pad_idx,
        smoothing=0.1,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(
        self,
        logits,
        target,
    ):

        log_probs = torch.log_softmax(
            logits,
            dim=-1,
        )

        with torch.no_grad():

            true_dist = torch.zeros_like(log_probs)

            true_dist.fill_(
                self.smoothing
                / (self.vocab_size - 2)
            )

            true_dist.scatter_(
                1,
                target.unsqueeze(1),
                self.confidence,
            )

            true_dist[:, self.pad_idx] = 0

            mask = target == self.pad_idx

            true_dist[mask] = 0

        loss = torch.sum(
            -true_dist * log_probs,
            dim=-1,
        )

        non_pad_mask = target != self.pad_idx

        loss = loss.masked_select(non_pad_mask)

        return loss.mean()


# ══════════════════════════════════════════════════════════════════════
# TRAIN / VALIDATION EPOCH
# ══════════════════════════════════════════════════════════════════════

def run_epoch(
    data_iter,
    model,
    loss_fn,
    optimizer,
    scheduler,
    epoch_num,
    is_train,
    device,
    log_gradients=False,
):

    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0
    total_confidence = 0
    num_batches = 0

    loop = tqdm(data_iter)

    for step, (src, tgt) in enumerate(loop):

        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:, :-1]
        tgt_y = tgt[:, 1:]

        src_mask = make_src_mask(
            src,
            model.pad_idx,
        ).to(device)

        tgt_mask = make_tgt_mask(
            tgt_input,
            model.pad_idx,
        ).to(device)

        with torch.set_grad_enabled(is_train):

            output = model(
                src,
                tgt_input,
                src_mask,
                tgt_mask,
            )

            logits = output.reshape(
                -1,
                output.size(-1),
            )

            targets = tgt_y.reshape(-1)

            loss = loss_fn(
                logits,
                targets,
            )

            probs = torch.softmax(
                logits,
                dim=-1,
            )

            target_probs = probs.gather(
                1,
                targets.unsqueeze(1),
            ).squeeze(1)

            mask = targets != model.pad_idx

            confidence = (
                target_probs[mask]
                .mean()
                .item()
            )

            total_confidence += confidence

            if is_train:

                optimizer.zero_grad()

                loss.backward()

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    1.0,
                )

                if (
                    log_gradients
                    and step < 1000
                ):

                    q_grad = None
                    k_grad = None

                    for name, param in model.named_parameters():

                        if (
                            "Wq.weight" in name
                            and param.grad is not None
                        ):
                            q_grad = (
                                param.grad.norm()
                                .item()
                            )

                        if (
                            "Wk.weight" in name
                            and param.grad is not None
                        ):
                            k_grad = (
                                param.grad.norm()
                                .item()
                            )

                    wandb.log(
                        {
                            "query_grad_norm": q_grad,
                            "key_grad_norm": k_grad,
                            "total_grad_norm": grad_norm.item(),
                        }
                    )

                optimizer.step()

                if scheduler is not None:
                    scheduler.step()

        total_loss += loss.item()
        num_batches += 1

        loop.set_description(
            f"Epoch {epoch_num}"
        )

        loop.set_postfix(
            loss=loss.item()
        )

    avg_loss = total_loss / num_batches

    avg_confidence = (
        total_confidence / num_batches
    )

    return avg_loss, avg_confidence


# ══════════════════════════════════════════════════════════════════════
# GREEDY DECODING
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def greedy_decode(
    model,
    src,
    src_mask,
    max_len,
    start_symbol,
):

    device = src.device

    memory = model.encode(
        src,
        src_mask,
    )

    ys = torch.ones(
        1,
        1,
        dtype=torch.long,
        device=device,
    ).fill_(start_symbol)

    for _ in range(max_len - 1):

        tgt_mask = make_tgt_mask(
            ys,
            model.pad_idx,
        ).to(device)

        out = model.decode(
            memory,
            src_mask,
            ys,
            tgt_mask,
        )

        prob = out[:, -1]

        next_word = torch.argmax(
            prob,
            dim=-1,
        ).item()

        ys = torch.cat(
            [
                ys,
                torch.ones(
                    1,
                    1,
                    dtype=torch.long,
                    device=device,
                ).fill_(next_word),
            ],
            dim=1,
        )

        if next_word == model.eos_idx:
            break

    return ys


# ══════════════════════════════════════════════════════════════════════
# BLEU EVALUATION
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_bleu(
    model,
    test_dataloader,
    tgt_vocab,
    device,
):

    model.eval()

    references = []
    hypotheses = []

    for src, tgt in tqdm(test_dataloader):

        src = src.to(device)

        for i in range(src.size(0)):

            src_i = src[i].unsqueeze(0)

            src_mask = make_src_mask(
                src_i,
                model.pad_idx,
            ).to(device)

            pred = greedy_decode(
                model,
                src_i,
                src_mask,
                max_len=100,
                start_symbol=tgt_vocab.stoi["<sos>"],
            )

            pred_tokens = []

            for idx in pred.squeeze().tolist():

                token = tgt_vocab.itos[idx]

                if token in [
                    "<sos>",
                    "<eos>",
                    "<pad>",
                ]:
                    continue

                pred_tokens.append(token)

            tgt_tokens = []

            for idx in tgt[i].tolist():

                token = tgt_vocab.itos[idx]

                if token in [
                    "<sos>",
                    "<eos>",
                    "<pad>",
                ]:
                    continue

                tgt_tokens.append(token)

            hypotheses.append(pred_tokens)
            references.append([tgt_tokens])

    bleu = corpus_bleu(
        references,
        hypotheses,
    )

    return bleu * 100


# ══════════════════════════════════════════════════════════════════════
# CHECKPOINTS
# ══════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    path,
):

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True,
    )

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict":
                scheduler.state_dict()
                if scheduler is not None
                else None,
        },
        path,
    )


def load_checkpoint(
    path,
    model,
    optimizer=None,
    scheduler=None,
):

    ckpt = torch.load(
        path,
        map_location="cpu",
    )

    model.load_state_dict(
        ckpt["model_state_dict"]
    )

    if (
        optimizer is not None
        and "optimizer_state_dict" in ckpt
    ):

        optimizer.load_state_dict(
            ckpt["optimizer_state_dict"]
        )

    if (
        scheduler is not None
        and ckpt["scheduler_state_dict"] is not None
    ):

        scheduler.load_state_dict(
            ckpt["scheduler_state_dict"]
        )

    return ckpt["epoch"]


# ══════════════════════════════════════════════════════════════════════
# ATTENTION VISUALIZATION
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def log_attention_maps(
    model,
    sample_src,
    device,
):

    model.eval()

    src = torch.tensor(
        sample_src,
        dtype=torch.long,
    ).unsqueeze(0).to(device)

    src_mask = make_src_mask(
        src,
        model.pad_idx,
    ).to(device)

    x = (
        model.src_embedding(src)
        * (model.d_model ** 0.5)
    )

    x = model.pos_enc(x)

    last_layer = model.encoder.layers[-1]

    mha = last_layer.mha

    Q = mha.Wq(x)
    K = mha.Wk(x)
    V = mha.Wv(x)

    B = Q.size(0)

    Q = Q.view(
        B,
        -1,
        mha.num_heads,
        mha.d_k,
    ).transpose(1, 2)

    K = K.view(
        B,
        -1,
        mha.num_heads,
        mha.d_k,
    ).transpose(1, 2)

    V = V.view(
        B,
        -1,
        mha.num_heads,
        mha.d_k,
    ).transpose(1, 2)

    _, attn = model.encoder.layers[
        -1
    ].mha.forward(
        x,
        x,
        x,
        src_mask,
    ), None

    scores = (
        Q @ K.transpose(-2, -1)
    ) / (mha.d_k ** 0.5)

    attn = torch.softmax(
        scores,
        dim=-1,
    )

    for h in range(mha.num_heads):

        heatmap = attn[
            0,
            h,
        ].detach().cpu().numpy()

        wandb.log(
            {
                f"attention_head_{h}":
                wandb.Image(heatmap)
            }
        )


# ══════════════════════════════════════════════════════════════════════
# MAIN TRAINING
# ══════════════════════════════════════════════════════════════════════

def run_training_experiment(args):

    set_seed(args.seed)

    wandb.login()

    wandb.init(
        project=args.wandb_project,
        name=args.run_name,
        config=vars(args),
    )

    train_dataset = Multi30kDataset(
        split="train",
    )

    val_dataset = Multi30kDataset(
        split="validation",
        src_vocab=train_dataset.src_vocab,
        tgt_vocab=train_dataset.tgt_vocab,
    )

    test_dataset = Multi30kDataset(
        split="test",
        src_vocab=train_dataset.src_vocab,
        tgt_vocab=train_dataset.tgt_vocab,
    )

    pad_idx = train_dataset.src_vocab.stoi["<pad>"]

    collate_fn = CollateFn(pad_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
    )

    device = torch.device(args.device)

    model = Transformer(
        src_vocab_size=len(train_dataset.src_vocab),
        tgt_vocab_size=len(train_dataset.tgt_vocab),
        d_model=args.d_model,
        N=args.N,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        learned_positional_encoding=args.learned_positional_encoding,
    ).to(device)

    if args.fixed_lr:

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.98),
            eps=1e-9,
        )

        scheduler = None

    else:

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=1.0,
            betas=(0.9, 0.98),
            eps=1e-9,
        )

        scheduler = NoamScheduler(
            optimizer,
            d_model=args.d_model,
            warmup_steps=args.warmup_steps,
        )

    loss_fn = LabelSmoothingLoss(
        vocab_size=len(train_dataset.tgt_vocab),
        pad_idx=train_dataset.tgt_vocab.stoi["<pad>"],
        smoothing=args.label_smoothing,
    )

    best_bleu = 0

    for epoch in range(args.epochs):

        train_loss, train_conf = run_epoch(
            train_loader,
            model,
            loss_fn,
            optimizer,
            scheduler,
            epoch,
            True,
            device,
            log_gradients=args.log_gradients,
        )

        val_loss, val_conf = run_epoch(
            val_loader,
            model,
            loss_fn,
            None,
            None,
            epoch,
            False,
            device,
        )

        bleu = evaluate_bleu(
            model,
            test_loader,
            train_dataset.tgt_vocab,
            device,
        )

        wandb.log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "bleu": bleu,
                "train_confidence": train_conf,
                "val_confidence": val_conf,
                "learning_rate":
                    optimizer.param_groups[0]["lr"],
            }
        )

        log_attention_maps(
            model,
            train_dataset[0][0],
            device,
        )

        if bleu > best_bleu:

            best_bleu = bleu

            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                args.checkpoint_path,
            )

    print(f"Best BLEU: {best_bleu:.4f}")

    wandb.finish()


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--d_model",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--N",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--num_heads",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--d_ff",
        type=int,
        default=2048,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=4000,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--fixed_lr",
        action="store_true",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--log_gradients",
        action="store_true",
    )

    parser.add_argument(
        "--wandb_project",
        type=str,
        default="da6401-a3",
    )

    parser.add_argument(
        "--run_name",
        type=str,
        default="transformer_run",
    )

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="checkpoints/best_model.pt",
    )
    parser.add_argument(
        "--learned_positional_encoding",
        
        action = "store_true",
    )

    args = parser.parse_args()

    run_training_experiment(args)