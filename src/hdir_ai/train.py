import argparse
import gc
import math
import time
from datetime import datetime
from pathlib import Path
import torch
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from rich.prompt import FloatPrompt, Prompt
from rich.table import Table

from .tokenizer import BPETokenizer, DataLoader
from .model import GPTConfig, MiniLM

console = Console()

torch.set_num_threads(2)
torch.set_num_interop_threads(2)

@torch.no_grad()
def estimate_loss(model: MiniLM, loader: DataLoader, eval_iters: int, device: str) -> dict[str, float]:
    """Average loss over multiple batches for stable train/val metrics."""
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = loader.get_batch(split, device=device)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    """Linear warmup followed by cosine decay."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

def select_run_dir(runs_dir: Path, mode: str | None = None, run_name: str | None = None) -> tuple[Path, bool]:
    """Choose between creating a new run folder or resuming an existing one."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    existing_runs = sorted([p for p in runs_dir.iterdir() if p.is_dir() and ((p / "latest.pt").exists() or (p / "checkpoint.pt").exists())])

    if mode is None:
        if existing_runs:
            console.print("[bold cyan]Training Mode:[/bold cyan]")
            console.print("  [green][1][/green] Start a new training run")
            console.print("  [yellow][2][/yellow] Continue a previous run")
            choice = Prompt.ask("Select option", choices=["1", "2"], default="1")
            is_resume = choice == "2"
        else:
            is_resume = False
    else:
        is_resume = mode == "resume"

    if not is_resume:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = run_name if run_name else f"run_{timestamp}"
        run_dir = runs_dir / folder_name
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir, False

    if not existing_runs:
        raise FileNotFoundError(f"No previous runs found in '{runs_dir}/' to resume.")

    if run_name:
        run_dir = runs_dir / run_name
        if not run_dir.exists():
            raise  FileNotFoundError(f"Run folder '{run_dir}' does not exist.")
        return run_dir, True

    console.print("\n[bold yellow]Available Runs to Resume:[/bold yellow]")
    for idx, folder in enumerate(existing_runs, start=1):
        ckpt_file = folder / "latest.pt" if (folder / "latest.pt").exists() else folder / "checkpoint.pt"
        meta = torch.load(ckpt_file, map_location="cpu", weights_only=False)
        saved_step = meta.get("step", "?")
        saved_loss = meta.get("val_loss", float("nan"))
        console.print(f"  [cyan][{idx}][/cyan] {folder.name} (step: {saved_step}, val_loss: {saved_loss:.4f})")

    pick = Prompt.ask("Enter run number to continue", choices=[str(i) for i in range(1, len(existing_runs) + 1)], default=str(len(existing_runs)))
    return existing_runs[int(pick) - 1], True

def load_or_encode_tokens(data_path: str, tokenizer: BPETokenizer, cache_path: Path) -> torch.Tensor:
    if cache_path.exists():
        console.print(f"[cyan]Loading cached token tensor from {cache_path}...[/cyan]")
        return torch.load(cache_path, weights_only=True)

    console.print("[cyan]Encoding dataset in memory-safe 1MB chunks...[/cyan]")
    token_chunks: list[torch.Tensor] = []
    with open(data_path, "r", encoding="utf-8") as f:
        while chunk := f.read(1_000_000):
            token_chunks.append(torch.tensor(tokenizer.encode(chunk), dtype=torch.int32))

    data_tensor = torch.cat(token_chunks)
    del token_chunks
    gc.collect()
    torch.save(data_tensor, cache_path)
    return data_tensor

def train(
        data_path: str = "data/input.txt",
        runs_dir: str = "runs",
        mode: str | None = None,
        run_name: str | None = None,
        vocab_size: int = 2048,
        batch_size: int = 32,
        grad_accum_steps: int = 2,
        block_size: int = 256,
        max_hours: float | None = None,
        est_steps_per_hour: int = 386,
        eval_interval: int = 500,
        eval_iters: int = 15,
        max_lr: float = 2e-4,
        min_lr: float = 3e-5,
        n_embd: int = 256,
        n_head: int = 8,
        n_layer: int = 6,
        dropout: float = 0.17
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_dir, is_resume = select_run_dir(Path(runs_dir), mode=mode, run_name=run_name)
    if max_hours is None:
        max_hours = FloatPrompt.ask("[bold cyan]Training duration (hours)[/bold cyan]", default=4.0)

    checkpoint_path = run_dir / "checkpoint.pt"
    latest_ckpt_path = run_dir / "latest.pt"
    tokenizer_path = run_dir / "tokenizer.json"

    console.print("[cyan]Loading dataset...[/cyan]")
    token_cache_path = run_dir / "tokens.pt"

    if is_resume:
        resume_file = latest_ckpt_path if latest_ckpt_path.exists() else checkpoint_path
        ckpt = torch.load(resume_file, map_location=device, weights_only=False)
        _tokenizer = BPETokenizer(tokenizer_path=str(tokenizer_path))
        config = ckpt["config"]
        config.dropout = 0.15
        start_step = ckpt.get("step", 0)
        best_val_loss = ckpt.get("val_loss", float("inf"))
        effective_max_lr = min(max_lr, 2.5e-4)
        effective_min_lr = min(min_lr, 3e-5)
        console.print(
            f"[bold yellow]Resuming '{run_dir.name}' from step {start_step} (best val_loss: {best_val_loss:.4f})[/bold yellow]")
    else:
        console.print("[cyan]Training BPE tokenizer on dataset sample...[/cyan]")
        with open(data_path, "r", encoding="utf-8") as f:
            sample_text = f.read(15_000_000)
        _tokenizer = BPETokenizer(text=sample_text, vocab_size=vocab_size)
        _tokenizer.save(str(tokenizer_path))
        del sample_text
        gc.collect()

        config = GPTConfig(
            vocab_size=_tokenizer.vocab_size,
            block_size=block_size,
            n_embd=n_embd,
            n_head=n_head,
            n_layer=n_layer,
            dropout=0.15
        )
        start_step = 0
        best_val_loss = float("inf")
        effective_max_lr = max_lr
        effective_min_lr = min_lr

    data_tensor = load_or_encode_tokens(data_path, _tokenizer, token_cache_path)
    loader = DataLoader(data_tensor, batch_size=batch_size, block_size=config.block_size)

    raw_model = MiniLM(config).to(device)
    if is_resume:
        raw_model.load_state_dict(ckpt["model_state_dict"])

    model = torch.compile(raw_model)
    param_count = sum(p.numel() for p in raw_model.parameters())

    session_steps = max(1, int(max_hours * est_steps_per_hour))
    warmup_steps = 0 if is_resume else min(300, session_steps // 10)

    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, betas=(0.9, 0.95), weight_decay=0.1)
    if is_resume and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if is_resume:
            del ckpt
            gc.collect()

    console.print(
        Panel.fit(
            f"[bold cyan]MiniLM Deep Story Training ({'Resume' if is_resume else 'New'}: {run_dir.name})[/bold cyan]\n"
            f"Device: [yellow]{device} (2 threads)[/yellow] | Params: [green]{param_count:,}[/green] | Start Step: [cyan]{start_step}[/cyan]\n"
            f"BPE Tokens: [green]{len(data_tensor):,}[/green] | Vocab: [green]{_tokenizer.vocab_size}[/green] | Context: [green]{config.block_size}[/green]\n"
            f"Architecture: [magenta]{config.n_layer}L / {config.n_head}H / {config.n_embd}D[/magenta] | Effective Batch: [green]{batch_size * grad_accum_steps}[/green]",
            border_style="cyan",
        )
    )

    table = Table(title=f"{max_hours}-Hour Training Log ({run_dir.name})", show_header=True, header_style="bold magenta")
    table.add_column("Step", justify="right", style="cyan")
    table.add_column("Elapsed", justify="right", style="dim")
    table.add_column("Train Loss", justify="right", style="green")
    table.add_column("Val Loss", justify="right", style="yellow")
    table.add_column("LR", justify="right", style="dim")
    table.add_column("Speed", justify="right", style="blue")
    table.add_column("Status", justify="center")

    start_time = time.perf_counter()
    last_eval_time = start_time
    max_seconds = max_hours * 3600

    with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("• [cyan]{task.fields[loss]}[/cyan]"),
            TimeRemainingColumn(),
            console=console
    ) as progress:
        task = progress.add_task(f"Training ({max_hours} hours)...", total=int(max_seconds), loss="loss: ---")

        step = start_step
        while True:
            elapsed = time.perf_counter() - start_time
            steps_done_this_session = step - start_step

            if steps_done_this_session % eval_interval == 0 or elapsed >= max_seconds:
                now = time.perf_counter()
                it_per_sec = eval_interval / max(now - last_eval_time, 1e-5) if steps_done_this_session > 0 else 0.0

                losses = estimate_loss(model, loader, eval_iters, device)

                state_payload = {
                    "model_state_dict": raw_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                    "tokenizer_path": str(tokenizer_path),
                    "step": step,
                    "val_loss": losses["val"],
                }
                torch.save(state_payload, latest_ckpt_path)

                status_badge = "[dim]SAVED LATEST[/dim]"
                if losses["val"] < best_val_loss:
                    best_val_loss = losses["val"]
                    status_badge = "[bold green]BEST SAVED[/bold green]"
                    state_payload["val_loss"] = best_val_loss
                    torch.save(state_payload, checkpoint_path)

                mins, secs = divmod(int(elapsed), 60)
                hrs, mins = divmod(mins, 60)
                lr = get_lr(steps_done_this_session, warmup_steps, session_steps, effective_max_lr, effective_min_lr)

                table.add_row(
                    f"{step}",
                    f"{hrs:02d}:{mins:02d}:{secs:02d}",
                    f"{losses['train']:.4f}",
                    f"{losses['val']:.4f}",
                    f"{lr:.1e}",
                    f"{it_per_sec:.2f} it/s" if steps_done_this_session > 0 else "-",
                    status_badge,
                )
                console.print(table)
                last_eval_time = time.perf_counter()

                if elapsed >= max_seconds:
                    console.print(
                        f"[bold yellow]Reached {max_hours}-hour time budget at step {step}. Stopping cleanly.[/bold yellow]")
                    break

            lr = get_lr(steps_done_this_session, warmup_steps, session_steps, effective_max_lr, effective_min_lr)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0
            for _ in range(grad_accum_steps):
                xb, yb = loader.get_batch("train", device=device)
                _, loss = model(xb, yb)
                loss = loss / grad_accum_steps
                loss.backward()
                accum_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            step += 1
            new_elapsed = min(int(time.perf_counter() - start_time), int(max_seconds))
            progress.update(task, completed=new_elapsed, loss=f"loss: {accum_loss:.4f} | step: {step}")

    best_ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    raw_model.load_state_dict(best_ckpt["model_state_dict"])
    raw_model.eval()
    prompt_text = "Once upon a time, "
    start_tokens = torch.tensor([_tokenizer.encode(prompt_text)], dtype=torch.long, device=device)
    generated_ids = raw_model.generate(
        start_tokens,
        max_new_tokens=220,
        tempreture=0.75,
        top_k=35,
        eos_token_id=_tokenizer.eot_id,
    )[0].tolist()

    console.print(
        Panel(
            _tokenizer.decode(generated_ids).replace("œ", '"'),
            title=f"[bold green]Generated Story Sample ({run_dir.name})[/bold green]",
            border_style="green"
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MiniLM (new run or resume previous run)")
    parser.add_argument("--mode", choices=["new", "resume"], default=None, help="Start a new run or resume an existing run")
    parser.add_argument("--run", type=str, default=None, help="Run folder name inside runs/")
    parser.add_argument("--hours", type=float, default=None, help="Training time budget in hours")
    args = parser.parse_args()

    train(mode=args.mode, run_name=args.run, max_hours=args.hours)
