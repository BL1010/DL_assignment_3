import os
import pickle

from collections import Counter

import torch

from torch.utils.data import Dataset

from datasets import load_dataset

import spacy


SPECIAL_TOKENS = [
    "<unk>",
    "<pad>",
    "<sos>",
    "<eos>",
]


# ══════════════════════════════════════════════════════════════════════
# VOCABULARY
# ══════════════════════════════════════════════════════════════════════

class Vocabulary:

    def __init__(self):

        self.itos = list(SPECIAL_TOKENS)

        self.stoi = {
            tok: idx
            for idx, tok in enumerate(self.itos)
        }

    def build_vocab(
        self,
        sentences,
        min_freq=2,
    ):

        counter = Counter()

        for sentence in sentences:
            counter.update(sentence)

        for word, freq in counter.items():

            if (
                freq >= min_freq
                and word not in self.stoi
            ):

                self.stoi[word] = len(self.itos)

                self.itos.append(word)

    def numericalize(self, tokens):

        unk_idx = self.stoi["<unk>"]

        return [
            self.stoi.get(token, unk_idx)
            for token in tokens
        ]

    def decode(self, ids):

        return [
            self.itos[idx]
            for idx in ids
        ]

    def __len__(self):

        return len(self.itos)


# ══════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════

class Multi30kDataset(Dataset):

    def __init__(
        self,
        split="train",
        src_vocab=None,
        tgt_vocab=None,
        max_len=100,
        cache_dir="artifacts",
    ):

        self.split = split
        self.max_len = max_len

        os.makedirs(
            cache_dir,
            exist_ok=True,
        )

        self.cache_path = os.path.join(
            cache_dir,
            f"{split}_cache.pkl",
        )

        self.vocab_path = os.path.join(
            cache_dir,
            "vocabs.pkl",
        )

        # ============================================================
        # LOAD DATASET
        # ============================================================

        self.dataset = load_dataset(
            "bentrevett/multi30k",
            split=split,
        )

        # ============================================================
        # LOAD TOKENIZERS
        # ============================================================

        self.de_tokenizer = spacy.load(
            "de_core_news_sm"
        )

        self.en_tokenizer = spacy.load(
            "en_core_web_sm"
        )

        # ============================================================
        # LOAD / BUILD VOCABS
        # ============================================================

        if (
            src_vocab is not None
            and tgt_vocab is not None
        ):

            self.src_vocab = src_vocab
            self.tgt_vocab = tgt_vocab

        else:

            if os.path.exists(self.vocab_path):

                with open(
                    self.vocab_path,
                    "rb",
                ) as f:

                    vocab_data = pickle.load(f)

                self.src_vocab = vocab_data[
                    "src_vocab"
                ]

                self.tgt_vocab = vocab_data[
                    "tgt_vocab"
                ]

            else:

                self.build_vocab()

        # ============================================================
        # LOAD CACHE IF EXISTS
        # ============================================================

        if os.path.exists(self.cache_path):

            print(
                f"Loading cached {split} dataset..."
            )

            with open(
                self.cache_path,
                "rb",
            ) as f:

                self.data = pickle.load(f)

        else:

            print(
                f"Processing {split} dataset..."
            )

            self.process_data()

            with open(
                self.cache_path,
                "wb",
            ) as f:

                pickle.dump(
                    self.data,
                    f,
                )

            print(
                f"Saved cache to {self.cache_path}"
            )

    # ══════════════════════════════════════════════════════════════
    # TOKENIZATION
    # ══════════════════════════════════════════════════════════════

    def tokenize_de(self, text):

        return [
            tok.text.lower()
            for tok in self.de_tokenizer(text)
        ]

    def tokenize_en(self, text):

        return [
            tok.text.lower()
            for tok in self.en_tokenizer(text)
        ]

    # ══════════════════════════════════════════════════════════════
    # BUILD VOCAB
    # ══════════════════════════════════════════════════════════════

    def build_vocab(self):

        print("Building vocabularies...")

        src_sentences = []
        tgt_sentences = []

        for sample in self.dataset:

            de_tokens = self.tokenize_de(
                sample["de"]
            )

            en_tokens = self.tokenize_en(
                sample["en"]
            )

            src_sentences.append(de_tokens)

            tgt_sentences.append(en_tokens)

        self.src_vocab = Vocabulary()

        self.tgt_vocab = Vocabulary()

        self.src_vocab.build_vocab(
            src_sentences
        )

        self.tgt_vocab.build_vocab(
            tgt_sentences
        )

        with open(
            self.vocab_path,
            "wb",
        ) as f:

            pickle.dump(
                {
                    "src_vocab":
                        self.src_vocab,
                    "tgt_vocab":
                        self.tgt_vocab,
                },
                f,
            )

        print(
            f"Saved vocabs to {self.vocab_path}"
        )

    # ══════════════════════════════════════════════════════════════
    # PROCESS DATA
    # ══════════════════════════════════════════════════════════════

    def process_data(self):

        self.data = []

        sos_src = self.src_vocab.stoi["<sos>"]
        eos_src = self.src_vocab.stoi["<eos>"]

        sos_tgt = self.tgt_vocab.stoi["<sos>"]
        eos_tgt = self.tgt_vocab.stoi["<eos>"]

        for sample in self.dataset:

            src_tokens = self.tokenize_de(
                sample["de"]
            )

            tgt_tokens = self.tokenize_en(
                sample["en"]
            )

            src = (
                [sos_src]
                + self.src_vocab.numericalize(
                    src_tokens
                )
                + [eos_src]
            )

            tgt = (
                [sos_tgt]
                + self.tgt_vocab.numericalize(
                    tgt_tokens
                )
                + [eos_tgt]
            )

            if (
                len(src) <= self.max_len
                and len(tgt) <= self.max_len
            ):

                self.data.append(
                    (
                        src,
                        tgt,
                    )
                )

    # ══════════════════════════════════════════════════════════════
    # PYTORCH DATASET
    # ══════════════════════════════════════════════════════════════

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        return self.data[idx]


# ══════════════════════════════════════════════════════════════════════
# COLLATE FUNCTION
# ══════════════════════════════════════════════════════════════════════

class CollateFn:

    def __init__(self, pad_idx):

        self.pad_idx = pad_idx

    def __call__(self, batch):

        src_batch = []
        tgt_batch = []

        src_max = max(
            len(x[0]) for x in batch
        )

        tgt_max = max(
            len(x[1]) for x in batch
        )

        for src, tgt in batch:

            src = (
                src
                + [self.pad_idx]
                * (src_max - len(src))
            )

            tgt = (
                tgt
                + [self.pad_idx]
                * (tgt_max - len(tgt))
            )

            src_batch.append(src)
            tgt_batch.append(tgt)

        return (
            torch.tensor(
                src_batch,
                dtype=torch.long,
            ),
            torch.tensor(
                tgt_batch,
                dtype=torch.long,
            ),
        )