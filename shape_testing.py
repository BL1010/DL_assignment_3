import torch

from dataset import Multi30kDataset, CollateFn
from torch.utils.data import DataLoader

from model import (
    Transformer,
    make_src_mask,
    make_tgt_mask,
)

dataset = Multi30kDataset(split="train")

loader = DataLoader(
    dataset,
    batch_size=2,
    collate_fn=CollateFn(
        dataset.src_vocab.stoi["<pad>"]
    )
)

src, tgt = next(iter(loader))

model = Transformer(
    src_vocab_size=len(dataset.src_vocab),
    tgt_vocab_size=len(dataset.tgt_vocab),
)

tgt_input = tgt[:, :-1]

src_mask = make_src_mask(src)
tgt_mask = make_tgt_mask(tgt_input)

out = model(
    src,
    tgt_input,
    src_mask,
    tgt_mask,
)

print(out.shape)