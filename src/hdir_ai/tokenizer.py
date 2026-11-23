import torch
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers


class BPETokenizer:
    """Fast subword tokenizer"""

    def __init__(self, text: str | None = None, vocab_size: int = 4096, tokenizer_path: str | None = None):
        if tokenizer_path:
            self.tokenizer = Tokenizer.from_file(tokenizer_path)
        elif text is not None:
            self.tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
            self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            self.tokenizer.decoder = decoders.ByteLevel()

            trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["<|endoftext|>", "<|unk|>"])
            self.tokenizer.train_from_iterator([text], trainer=trainer)
        else:
            raise ValueError("Provide either `text` to train or `tokenizer_path` to load.")

        self.vocab_size = self.tokenizer.get_vocab_size()
        self.eot_id = self.tokenizer.token_to_id("<|endoftext|>")

    def encode(self, s: str) -> list[int]:
        return self.tokenizer.encode(s).ids

    def decode(self, tokens: list[int] | list[list[int]]) -> str:
        if tokens and isinstance(tokens[0], list):
            tokens = tokens[0]
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def save(self, path: str):
        self.tokenizer.save(path)


class DataLoader:
    def __init__(self, data: torch.Tensor, batch_size: int, block_size: int, split_ratio: float = 0.9):
        self.batch_size = batch_size
        self.block_size = block_size

        n = int(split_ratio * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]

    def get_batch(self, split: str, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
        source = self.train_data if split == "train" else self.val_data
        ix = torch.randint(len(source) - self.block_size - 1, (self.batch_size,))
        x = torch.stack([source[i : i + self.block_size] for i in ix]).long()
        y = torch.stack([source[i + 1 : i + self.block_size + 1] for i in ix]).long()
        return x.to(device), y.to(device)
