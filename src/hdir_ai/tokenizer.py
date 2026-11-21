import torch

class CharTokenizer:
    def __init__(self, text: str):
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)

        self.str_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_str = {i: ch for i, ch in enumerate(chars)}

    def encode(self, s: str) -> list[int]:
        """Convert a str into a list of int token IDs."""
        return [self.str_to_idx[c] for c in s]

    def decode(self, tokens: list[int]) -> str:
        """Convert a list of int token IDs back into a str."""
        return "".join(self.idx_to_str[i] for i in tokens)

class DataLoader:
    def __init__(self, data: torch.Tensor, batch_size: int, block_size: int, split_ratio: float = 0.9):
        self.batch_size = batch_size
        self.block_size = block_size

        n = int(split_ratio * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]

    def get_batch(self, split: str, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
        """Generate a random batch of inputs (x) and shifted targets (y)."""
        source = self.train_data if split == "train" else self.val_data

        ix = torch.randint(len(source) - self.block_size, (self.batch_size,))

        x = torch.stack([source[i : i + self.block_size] for i in ix])
        y = torch.stack([source[i + 1 : i + self.block_size + 1] for i in ix])

        return x.to(device), y.to(device)