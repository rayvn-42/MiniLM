import argparse
import time
from pathlib import Path
import torch
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.prompt import Prompt

from .tokenizer import BPETokenizer
from .model import MiniLM

console = Console()


def choose_checkpoint(runs_dir: str = "runs", explicit_path: str | None = None) -> Path:
    if explicit_path:
        return Path(explicit_path)

    runs_path = Path(runs_dir)
    candidates = sorted(runs_path.glob("*/checkpoint.pt"), key=lambda p: p.stat().st_mtime) if runs_path.exists() else []

    if not candidates:
        raise FileNotFoundError(f"No checkpoint.pt found inside '{runs_dir}/'.")

    if len(candidates) == 1:
        return candidates[0]

    console.print("\n[bold cyan]Available Trained Models:[/bold cyan]")
    for idx, ckpt_path in enumerate(candidates, start=1):
        meta = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        step = meta.get("step", "?")
        val_loss = meta.get("val_loss", float("nan"))
        console.print(f"  [cyan][{idx}][/cyan] {ckpt_path.parent.name} (step: {step}, val_loss: {val_loss:.4f})")

    pick = Prompt.ask(
        "Select model number",
        choices=[str(i) for i in range(1, len(candidates) + 1)],
        default=str(len(candidates)),
    )
    return candidates[int(pick) - 1]


def load_model(checkpoint_path: Path, device: str = "cpu") -> tuple[MiniLM, BPETokenizer]:
    tok_path = checkpoint_path.parent / "tokenizer.json"
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    tokenizer = BPETokenizer(tokenizer_path=str(tok_path))

    model = MiniLM(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, tokenizer


@torch.no_grad()
def stream_story(
    model: MiniLM,
    tokenizer: BPETokenizer,
    prompt: str,
    max_new_tokens: int = 250,
    temperature: float = 0.7,
    top_k: int = 35,
    repetition_penalty: float = 1.15,
    device: str = "cpu",
):
    input_ids = tokenizer.encode(prompt)
    idx = torch.tensor([input_ids], dtype=torch.long, device=device)

    with Live(Panel(prompt, title="[bold green]MiniLM Story[/bold green]", border_style="green"), refresh_per_second=15) as live:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -model.config.block_size :]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)

            for token_id in set(idx[0].tolist()):
                if logits[0, token_id] < 0:
                    logits[0, token_id] *= repetition_penalty
                else:
                    logits[0, token_id] /= repetition_penalty

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

            if tokenizer.eot_id is not None and idx_next.item() == tokenizer.eot_id:
                break

            current_text = tokenizer.decode(idx[0].tolist()).replace("œ", '"')
            live.update(Panel(current_text, title="[bold green]MiniLM Story[/bold green]", border_style="green"))
            time.sleep(0.01)


def main():
    parser = argparse.ArgumentParser(description="Generate stories with MiniLM")
    parser.add_argument("--checkpoint", type=str, default=None, help="Direct path to a checkpoint.pt file")
    parser.add_argument("--temp", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--tokens", type=int, default=250, help="Max new tokens to generate")
    parser.add_argument("--penalty", type=float, default=1.15, help="Repetition penalty")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = choose_checkpoint(explicit_path=args.checkpoint)
    console.print(f"[bold cyan]Loading {ckpt_path} on {device}...[/bold cyan]")

    model, tokenizer = load_model(ckpt_path, device=device)
    param_count = sum(p.numel() for p in model.parameters())
    console.print(f"[green]{param_count:,} parameters loaded! Type a story prompt (or 'quit' to exit).[/green]\n")

    while True:
        prompt_text = Prompt.ask("[bold yellow]Prompt[/bold yellow]", default="Once upon a time, ")
        if prompt_text.strip().lower() in {"quit", "exit", "q"}:
            break

        stream_story(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt_text,
            max_new_tokens=args.tokens,
            temperature=args.temp,
            repetition_penalty=args.penalty,
            device=device,
        )
        console.print()


if __name__ == "__main__":
    main()
