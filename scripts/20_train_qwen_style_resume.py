import os
import math
import json
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def ddp_setup():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        return True, rank, local_rank, world_size
    return False, 0, 0, 1


class RandomMemmapDataset:
    def __init__(self, path, block_size, dtype=np.uint16, seed=42):
        self.path = path
        self.data = np.memmap(path, dtype=dtype, mode="r")
        self.block_size = block_size
        self.rng = np.random.default_rng(seed)
        self.max_start = len(self.data) - block_size - 1
        if self.max_start <= 0:
            raise RuntimeError(f"Dataset too small: {path}")

    def batch(self, batch_size, device):
        starts = self.rng.integers(0, self.max_start, size=batch_size)
        x = np.stack([np.asarray(self.data[s:s+self.block_size], dtype=np.int64) for s in starts])
        y = np.stack([np.asarray(self.data[s+1:s+self.block_size+1], dtype=np.int64) for s in starts])
        return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


def precompute_rope(block_size, head_dim, device, theta=1000000.0):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(block_size, device=device).float()
    freqs = torch.outer(t, inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    # x: B, H, T, D
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads):
        super().__init__()
        assert dim % n_heads == 0
        assert n_heads % n_kv_heads == 0

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.kv_repeat = n_heads // n_kv_heads

        self.q_proj = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, dim, bias=False)

    def forward(self, x, cos, sin):
        b, t, c = x.shape

        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos[:t], sin[:t])
        k = apply_rope(k, cos[:t], sin[:t])

        if self.kv_repeat != 1:
            k = k.repeat_interleave(self.kv_repeat, dim=1)
            v = v.repeat_interleave(self.kv_repeat, dim=1)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.o_proj(y)


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads, mlp_dim):
        super().__init__()
        self.input_norm = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, n_heads, n_kv_heads)
        self.post_norm = RMSNorm(dim)
        self.mlp = SwiGLU(dim, mlp_dim)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.input_norm(x), cos, sin)
        x = x + self.mlp(self.post_norm(x))
        return x


class QwenStyleLM(nn.Module):
    def __init__(self, vocab_size, dim, n_layers, n_heads, n_kv_heads, mlp_dim, block_size, tie_embeddings=False):
        super().__init__()
        self.block_size = block_size
        self.head_dim = dim // n_heads

        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            Block(dim, n_heads, n_kv_heads, mlp_dim)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        if tie_embeddings:
            self.lm_head.weight = self.embed.weight

    def forward(self, idx, targets=None):
        b, t = idx.shape
        device = idx.device

        x = self.embed(idx)
        cos, sin = precompute_rope(self.block_size, self.head_dim, device)

        for block in self.blocks:
            x = block(x, cos, sin)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )

        return logits, loss


def make_config(profile):
    if profile == "sanity":
        return {
            "dim": 384,
            "n_layers": 6,
            "n_heads": 6,
            "n_kv_heads": 2,
            "mlp_dim": 1024,
            "block_size": 512,
            "tie_embeddings": True,
        }

    if profile == "qwen08b":
        return {
            "dim": 1536,
            "n_layers": 24,
            "n_heads": 16,
            "n_kv_heads": 8,
            "mlp_dim": 4096,
            "block_size": 1024,
            "tie_embeddings": False,
        }

    raise ValueError(profile)


@torch.no_grad()
def evaluate(model, val_data, device, batch_size, eval_iters, ddp, rank):
    model.eval()
    losses = []

    raw_model = model.module if ddp else model

    for _ in range(eval_iters):
        x, y = val_data.batch(batch_size, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        losses.append(loss.detach())

    loss = torch.stack(losses).mean()

    if ddp:
        dist.all_reduce(loss, op=dist.ReduceOp.AVG)

    model.train()
    return loss.item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-bin", required=True)
    ap.add_argument("--val-bin", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--profile", choices=["sanity", "qwen08b"], default="sanity")
    ap.add_argument("--vocab-size", type=int, default=65536)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup-steps", type=int, default=100)
    ap.add_argument("--eval-interval", type=int, default=100)
    ap.add_argument("--eval-iters", type=int, default=20)
    ap.add_argument("--save-interval", type=int, default=500)
    ap.add_argument("--resume", default=None, help="Path to checkpoint to resume model weights from")
    args = ap.parse_args()

    ddp, rank, local_rank, world_size = ddp_setup()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    cfg = make_config(args.profile)
    block_size = cfg["block_size"]

    if rank == 0:
        print("config:", json.dumps(cfg, indent=2), flush=True)
        print("world_size:", world_size, flush=True)

    train_data = RandomMemmapDataset(args.train_bin, block_size, seed=1234 + rank)
    val_data = RandomMemmapDataset(args.val_bin, block_size, seed=5678 + rank)

    model = QwenStyleLM(
        vocab_size=args.vocab_size,
        dim=cfg["dim"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        n_kv_heads=cfg["n_kv_heads"],
        mlp_dim=cfg["mlp_dim"],
        block_size=cfg["block_size"],
        tie_embeddings=cfg["tie_embeddings"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=True)
        start_step = int(ckpt.get("step", 0))
        if rank == 0:
            print(f"RESUME_FROM: {args.resume}", flush=True)
            print(f"RESUME_STEP: {start_step}", flush=True)

    if rank == 0:
        print(f"parameters: {n_params:,}", flush=True)

    model = model.to(dtype=torch.bfloat16)

    if ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    out_dir = Path(args.out_dir)
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "run_config.json").write_text(json.dumps({
            "args": vars(args),
            "model_config": cfg,
            "params": n_params,
            "world_size": world_size,
        }, indent=2), encoding="utf-8")

    if ddp:
        dist.barrier()

    model.train()
    t0 = time.time()

    for step in range(start_step + 1, args.max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0

        for micro in range(args.grad_accum):
            x, y = train_data.batch(args.batch_size, device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(x, y)
                loss = loss / args.grad_accum

            loss.backward()
            loss_accum += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        if step <= args.warmup_steps:
            lr = args.lr * step / max(1, args.warmup_steps)
        else:
            progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
            lr = args.lr * 0.5 * (1.0 + math.cos(math.pi * progress))

        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.step()

        if rank == 0 and step % 10 == 0:
            dt = time.time() - t0
            tokens = step * args.batch_size * args.grad_accum * block_size * world_size
            print(
                f"step={step} loss={loss_accum:.4f} lr={lr:.2e} "
                f"tokens={tokens:,} tok/s={tokens/max(dt,1):.0f}",
                flush=True,
            )

        if step % args.eval_interval == 0:
            val_loss = evaluate(model, val_data, device, args.batch_size, args.eval_iters, ddp, rank)
            if rank == 0:
                ppl = math.exp(min(20, val_loss))
                print(f"EVAL step={step} val_loss={val_loss:.4f} ppl={ppl:.2f}", flush=True)

        if rank == 0 and step % args.save_interval == 0:
            raw_model = model.module if ddp else model
            ckpt = {
                "model": raw_model.state_dict(),
                "step": step,
                "config": cfg,
                "params": n_params,
            }
            torch.save(ckpt, out_dir / f"ckpt_step_{step}.pt")
            print(f"saved checkpoint step {step}", flush=True)

    if rank == 0:
        raw_model = model.module if ddp else model
        torch.save({
            "model": raw_model.state_dict(),
            "step": args.max_steps,
            "config": cfg,
            "params": n_params,
        }, out_dir / "final.pt")
        print("training done", flush=True)

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
