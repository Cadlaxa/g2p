import re
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

PAD_IDX, UNK_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ["<pad>", "<unk>", "<bos>", "<eos>"]


def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src, tgt in batch:
        src_batch.append(src)
        tgt_batch.append(tgt)

    src_lengths = torch.tensor([len(s) for s in src_batch], dtype=torch.int32)
    tgt_lengths = torch.tensor([len(s) for s in tgt_batch], dtype=torch.int32)
    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return (
        src_batch.transpose(0, 1).contiguous(),
        tgt_batch.transpose(0, 1).contiguous(),
        src_lengths,
        tgt_lengths,
    )


class SphinxDataset(Dataset):
    def __init__(
        self,
        path,
        graphemes: list[str],
        phonemes: list[str],
        comment_prefix=";;;",
        remove_word_digits=True,
        remove_phoneme_digits=True,
        lowercase_word=True,
    ):
        self.graphemes = graphemes
        assert len(self.graphemes) > 4
        for i in range(4):
            assert self.graphemes[i] == special_symbols[i]
        self.grapheme_indexs = {v: i for i, v in enumerate(self.graphemes)}

        self.phonemes = phonemes
        assert len(self.phonemes) > 4
        for i in range(4):
            assert self.phonemes[i] == special_symbols[i]
        self.phoneme_indexs = {v: i for i, v in enumerate(self.phonemes)}

        self.entries = self.load_dict(
            path,
            comment_prefix,
            remove_word_digits,
            remove_phoneme_digits,
            lowercase_word,
        )

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.entries[idx]

    def load_dict(
        self,
        path,
        comment_prefix,
        remove_word_digits,
        remove_phoneme_digits,
        lowercase_word,
    ):
        rm_digit = re.compile(r"\(\d+\)$|\d+$")
        entries = []
        ignored_graphemes = set()
        ignored_phonemes = set()

        with open(path, "r", encoding="utf8") as f:
            for line in f:
                line = line.strip()
                if line == "" or line.startswith(comment_prefix):
                    continue

                parts = line.split()
                if len(parts) < 2:
                    continue

                word = parts[0]
                pron = parts[1:]
                if remove_word_digits:
                    word = rm_digit.sub("", word)
                if lowercase_word:
                    word = word.lower()
                if remove_phoneme_digits:
                    pron = [rm_digit.sub("", p) for p in pron]

                for c in word:
                    if c not in self.grapheme_indexs:
                        ignored_graphemes.add(c)
                for p in pron:
                    if p not in self.phoneme_indexs:
                        ignored_phonemes.add(p)

                src_tensor = torch.tensor(
                    [BOS_IDX]
                    + [self.grapheme_indexs.get(s, UNK_IDX) for s in word]
                    + [EOS_IDX],
                    dtype=torch.int32,
                )
                tgt_tensor = torch.tensor(
                    [self.phoneme_indexs.get(s, UNK_IDX) for s in pron] + [EOS_IDX],
                    dtype=torch.int32,
                )
                entries.append((src_tensor, tgt_tensor))

        print("graphemes: {}".format(", ".join(self.graphemes)))
        print("ignored graphemes: {}".format(", ".join(sorted(ignored_graphemes))))
        print("phonemes: {}".format(", ".join(self.phonemes)))
        print("ignored phonemes: {}".format(", ".join(sorted(ignored_phonemes))))
        return entries