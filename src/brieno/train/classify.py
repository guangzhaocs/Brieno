"""Fine-tuning a classification head on top of the pretrained model.

One trainer serves every per-chunk classification task — barcode demultiplexing,
species identification — because they differ only in the data and the number of
classes, not in the architecture. The number of classes comes from the dataset's
``labels.json``, so nothing here needs to know which task it is running.

Several heads can be trained in one job. They share the frozen front-end, so comparing
"token ids alone" against "tokens through the pretrained transformer" costs one forward
pass over the tokenizer rather than one per head — which is what makes an honest
ablation affordable.
"""
import datetime
import gzip
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from brieno.data.encode import load_tokenizer
from brieno.downstream.data import (BalancedSampler, ChunkDataset, PartIterDataset,
                                    load_split, subsample)
from brieno.downstream.features import (FrozenFeatures, load_transformer,
                                        transformer_hidden)
from brieno.downstream.heads import Heads, RawEncoder
from brieno.train.utils import JsonlLogger

__all__ = ["ALL_HEADS", "make_features", "evaluate", "read_level", "finetune"]

ALL_HEADS = ["A", "B1", "B2", "C1", "C2", "D1"]
NEEDS_TRANSFORMER = ("C1", "C2")


def _autocast():
    return torch.autocast("cuda", dtype=torch.bfloat16,
                          enabled=torch.cuda.is_available() and torch.cuda.is_bf16_supported())


def make_features(x, feat, tf, heads, raw=None):
    """Raw chunks -> the input each head wants.

    The tokenizer runs once; the transformer runs once more only if a C head is in
    play. C1 is computed without gradients even when C2 is training, so the two can
    share a job without the frozen head quietly being fine-tuned.
    """
    f = {}
    if feat is not None:
        lat, q, idx = feat(x)
        f.update({"A": lat, "B1": q, "B2": idx})
    if raw is not None:
        with _autocast():
            f["D1"] = raw(x)
    if "C1" in heads:
        with torch.no_grad(), _autocast():
            f["C1"] = transformer_hidden(tf, f["B2"]).float()
    if "C2" in heads:
        with _autocast():
            f["C2"] = transformer_hidden(tf, f["B2"]).float()
    return f


@torch.no_grad()
def evaluate(feat, tf, heads, model, loader, device, collect=False, raw=None):
    """Chunk-level accuracy per head; with ``collect``, also the per-chunk log-probs."""
    model.eval()
    tf_was_training = tf is not None and tf.training
    if tf is not None:
        tf.eval()
    if raw is not None:
        raw.eval()

    correct = {h: 0 for h in heads}
    n = 0
    store = {h: [] for h in heads}
    gids, poss, labs = [], [], []

    for x, y, gid, pos, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device)
        f = make_features(x, feat, tf, heads, raw)
        with _autocast():
            out = model(f)
        for h in heads:
            lp = F.log_softmax(out[h].float(), -1)
            correct[h] += (lp.argmax(-1) == y).sum().item()
            if collect:
                store[h].append(lp.half().cpu().numpy())
        n += len(y)
        if collect:
            gids.append(gid.numpy())
            poss.append(pos.numpy())
            labs.append(y.cpu().numpy())

    if tf_was_training:
        tf.train()
    if raw is not None:
        raw.train()

    acc = {h: correct[h] / max(n, 1) for h in heads}
    if not collect:
        return acc, None
    return acc, dict(logp={h: np.concatenate(store[h]) for h in heads},
                     gid=np.concatenate(gids), pos=np.concatenate(poss),
                     label=np.concatenate(labs))


def read_level(res, heads, n_classes, classes, runs=None):
    """Aggregate per-chunk log-probabilities into one call per read.

    Several aggregations are reported rather than one, because which is best depends on
    the task: a barcode sits in the first chunk, so ``first1`` is the honest number
    there, while a species call may use the whole read. Reporting all of them keeps the
    choice explicit instead of buried.
    """
    order = np.lexsort((res["pos"], res["gid"]))
    gid, pos, lab = res["gid"][order], res["pos"][order], res["label"][order]
    bounds = np.flatnonzero(np.r_[True, gid[1:] != gid[:-1], True])
    starts, ends = bounds[:-1], bounds[1:]
    read_label = lab[starts]
    n_chunks = ends - starts

    out = {}
    for h in heads:
        lp = res["logp"][h][order].astype(np.float32)
        p = np.exp(lp)
        pred_chunk = lp.argmax(-1)

        preds = {}
        mlp = np.stack([lp[s:e].mean(0) for s, e in zip(starts, ends)])
        preds["mean_logp"] = mlp.argmax(-1)
        preds["mean_prob"] = np.array([p[s:e].mean(0).argmax() for s, e in zip(starts, ends)])
        preds["majority"] = np.array([np.bincount(pred_chunk[s:e], minlength=n_classes).argmax()
                                      for s, e in zip(starts, ends)])
        for k in (1, 2, 4):
            preds[f"first{k}_mean_logp"] = np.array(
                [lp[s:min(s + k, e)].mean(0).argmax() for s, e in zip(starts, ends)])

        m = {"chunk_acc": float((pred_chunk == lab).mean()),
             "n_reads": int(len(starts)), "n_chunks": int(len(lab))}
        for name, pr in preds.items():
            m[f"read_acc_{name}"] = float((pr == read_label).mean())

        pr = preds["mean_logp"]
        # A class absent from this split has no recall to report. Writing NaN would
        # produce a metrics.json that strict JSON parsers reject, so report null.
        m["per_class_recall_mean_logp"] = {
            classes[c]: (float((pr[read_label == c] == c).mean())
                         if (read_label == c).any() else None)
            for c in range(n_classes)}
        m["confusion_mean_logp"] = [
            [int(((read_label == t) & (pr == c)).sum()) for c in range(n_classes)]
            for t in range(n_classes)]
        m["read_acc_by_nchunks_mean_logp"] = {
            str(k): float((pr[n_chunks == k] == read_label[n_chunks == k]).mean())
            for k in sorted(set(n_chunks.tolist())) if (n_chunks == k).any()}
        if runs is not None:
            row = gid[starts] >> 24
            m["per_run_recall_mean_logp"] = {
                runs[i]["run"]: dict(
                    n_reads=int((row == i).sum()),
                    recall=(float((pr[row == i] == read_label[row == i]).mean())
                            if (row == i).any() else None))
                for i in sorted(set(row.tolist()))}
        out[h] = (m, preds["mean_logp"], read_label, gid[starts])
    return out


def _write_per_read(path, gids, truth, pred, classes):
    with gzip.open(path, "wt") as fh:
        fh.write("read_gid\ttrue\tpred\n")
        for g, t, p in zip(gids, truth, pred):
            fh.write(f"{g}\t{classes[t]}\t{classes[p]}\n")


def finetune(root, setting, fold, tokenizer_ckpt, transformer_ckpt=None, heads=("C1",),
             out_root=".", task_name="classify", run_name=None, epochs=12,
             per_class=50000, val_chunks=50000, batch=128, eval_batch=512, lr=5e-4,
             tf_lr=5e-5, weight_decay=0.05, accum=1, workers=8, seed=0, resume=None,
             transformer_random=False):
    """Train one or more heads on a classification dataset. Returns the output directory."""
    heads = list(heads)
    unknown = [h for h in heads if h not in ALL_HEADS]
    if unknown:
        raise ValueError(f"unknown head(s) {unknown}; choose from {ALL_HEADS}")
    if "C1" in heads and "C2" in heads:
        raise ValueError("C1 and C2 cannot share a job: C2 fine-tunes the transformer "
                         "that C1 must see frozen. Run them separately.")
    if any(h in NEEDS_TRANSFORMER for h in heads) and transformer_ckpt is None:
        raise ValueError(f"heads {NEEDS_TRANSFORMER} need --transformer")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    split_dir = os.path.join(root, setting, f"fold{fold}")
    labels = json.load(open(os.path.join(root, setting, "labels.json"), encoding="utf-8"))
    classes = labels["classes"]
    n_classes = len(classes)

    name = run_name or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(out_root, f"{task_name}_{'+'.join(heads)}", name)
    os.makedirs(out_dir, exist_ok=True)
    history = JsonlLogger(os.path.join(out_dir, "history.jsonl"))
    print(f"writing to {out_dir}")
    print(f"{n_classes} classes: {', '.join(classes)}")

    # ---------------------------------------------------------------- data
    tr = load_split(os.path.join(split_dir, "train.tsv"), root)
    va = subsample(load_split(os.path.join(split_dir, "val.tsv"), root), val_chunks, seed)
    te = load_split(os.path.join(split_dir, "test.tsv"), root)
    print(f"train {len(tr['chunk_k']):,d} chunks | val {len(va['chunk_k']):,d} | "
          f"test {len(te['chunk_k']):,d} over {len(te['runs'])} runs")

    sampler = BalancedSampler(tr["label"], per_class, n_classes, seed)
    dl_tr = DataLoader(ChunkDataset(tr), batch_size=batch, sampler=sampler,
                       num_workers=workers, pin_memory=True, drop_last=True)
    dl_va = DataLoader(PartIterDataset(va), batch_size=eval_batch,
                       num_workers=workers, pin_memory=True)

    # ---------------------------------------------------------------- front-end
    tok, tok_cfg = load_tokenizer(tokenizer_ckpt, device=device)
    feat = FrozenFeatures(tok) if any(h != "D1" for h in heads) else None
    hidden_dim = int(tok_cfg["hidden_dim"])
    downsample = int(tok_cfg.get("downsample", 16))

    tf = tf_cfg = None
    tf_train = "C2" in heads
    if any(h in NEEDS_TRANSFORMER for h in heads):
        tf, tf_cfg = load_transformer(transformer_ckpt, device, trainable=tf_train,
                                      random_init=transformer_random)
        print(f"transformer: {tf_cfg['n_layer']} layers, d_model {tf_cfg['d_model']}, "
              f"{'fine-tuned' if tf_train else 'frozen'}"
              + ("  (randomly initialised control)" if transformer_random else ""))

    raw = RawEncoder(hidden_dim, downsample).to(device) if "D1" in heads else None

    model = Heads(heads, hidden_dim, int(tok_cfg["num_quantizers"]),
                  int(tok_cfg["num_tokens"]), n_classes,
                  ctx_dim=int(tf_cfg["d_model"]) if tf_cfg else None).to(device)
    for h in heads:
        print(f"  head {h}: {model.n_params(h) / 1e6:.2f} M parameters")

    # ---------------------------------------------------------------- optimisation
    groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": lr}]
    if tf_train:
        # The transformer is pretrained and the head is not, so they must not move at
        # the same rate: a head-sized step would undo what pretraining established.
        groups.append({"params": [p for p in tf.parameters() if p.requires_grad], "lr": tf_lr})
    if raw is not None:
        groups.append({"params": list(raw.parameters()), "lr": lr})

    opt = torch.optim.AdamW(groups, lr=lr, weight_decay=weight_decay)
    steps_per_epoch = len(dl_tr) // accum
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[g["lr"] for g in groups], total_steps=epochs * steps_per_epoch,
        pct_start=0.05)

    start_epoch = 1
    best = {h: (-1.0, None) for h in heads}
    if resume:
        state = torch.load(resume, map_location=device)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        if tf_train and "tf" in state:
            tf.load_state_dict(state["tf"])
        if raw is not None and "raw" in state:
            raw.load_state_dict(state["raw"])
        start_epoch = state["epoch"] + 1
        best = {h: tuple(v) for h, v in state["best"].items()}
        torch.set_rng_state(state["rng"]["torch"])
        np.random.set_state(state["rng"]["numpy"])
        print(f"resumed from {resume} at epoch {state['epoch']}")

    trainable = ([p for p in model.parameters() if p.requires_grad]
                 + (list(tf.parameters()) if tf_train else [])
                 + (list(raw.parameters()) if raw is not None else []))

    # ---------------------------------------------------------------- loop
    for epoch in range(start_epoch, epochs + 1):
        sampler.set_epoch(epoch)
        model.train()
        if tf_train:
            tf.train()
        if raw is not None:
            raw.train()

        running, seen, t0 = 0.0, 0, time.time()
        opt.zero_grad()
        for step, (x, y, _, _, _) in enumerate(dl_tr):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            f = make_features(x, feat, tf, heads, raw)
            with _autocast():
                out = model(f)
                loss = sum(F.cross_entropy(out[h].float(), y) for h in heads) / len(heads)
            (loss / accum).backward()

            if (step + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
            running += loss.item()
            seen += 1

        acc, _ = evaluate(feat, tf, heads, model, dl_va, device, raw=raw)
        history.log(split="val", epoch=epoch, loss=running / max(seen, 1),
                    **{f"acc_{h}": acc[h] for h in heads})
        print(f"epoch {epoch}/{epochs}  loss {running / max(seen, 1):.4f}  "
              + "  ".join(f"{h} {acc[h]:.4f}" for h in heads)
              + f"  ({time.time() - t0:.0f}s)")

        for h in heads:
            if acc[h] > best[h][0]:
                best[h] = (acc[h], epoch)
                torch.save({"head": model.head_state(h), "epoch": epoch, "val_acc": acc[h],
                            "classes": classes,
                            "tokenizer": os.path.basename(tokenizer_ckpt),
                            "transformer": os.path.basename(transformer_ckpt) if transformer_ckpt else None,
                            **({"transformer_state": tf.state_dict()} if h == "C2" else {}),
                            **({"raw_encoder": raw.state_dict()} if h == "D1" else {})},
                           os.path.join(out_dir, f"best_{h}.pt"))

        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": epoch,
                    "best": {h: list(v) for h, v in best.items()},
                    "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state()},
                    **({"tf": tf.state_dict()} if tf_train else {}),
                    **({"raw": raw.state_dict()} if raw is not None else {})},
                   os.path.join(out_dir, "last.pt"))

    # ---------------------------------------------------------------- test
    print("\nscoring the test split with each head's best epoch ...")
    dl_te = DataLoader(PartIterDataset(te), batch_size=eval_batch, num_workers=workers,
                       pin_memory=True)
    metrics = {"classes": classes, "task": task_name, "setting": setting, "fold": fold,
               "best_epoch": {h: best[h][1] for h in heads},
               "best_val_acc": {h: best[h][0] for h in heads}, "test": {}}

    for h in heads:
        state = torch.load(os.path.join(out_dir, f"best_{h}.pt"), map_location=device)
        model.load_head_state(h, state["head"])
        if h == "C2" and "transformer_state" in state:
            tf.load_state_dict(state["transformer_state"])
        if h == "D1" and "raw_encoder" in state:
            raw.load_state_dict(state["raw_encoder"])

        _, res = evaluate(feat, tf, [h], model, dl_te, device, collect=True, raw=raw)
        (m, pred, truth, gids), = read_level(res, [h], n_classes, classes, te["runs"]).values()
        metrics["test"][h] = m
        _write_per_read(os.path.join(out_dir, f"per_read_{h}.tsv.gz"), gids, truth, pred, classes)
        print(f"  {h}: chunk {m['chunk_acc']:.4f}  read {m['read_acc_mean_logp']:.4f} "
              f"(first1 {m['read_acc_first1_mean_logp']:.4f})  over {m['n_reads']:,d} reads")

    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    return out_dir
