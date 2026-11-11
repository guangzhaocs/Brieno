"""Fine-tuning a basecalling head on top of the pretrained model.

The head is a fixed sequence back-end with the CTC-CRF output layer of Dorado and
Bonito (:mod:`brieno.downstream.basecall`). It is trained on signal chunks paired with
the reference bases they align to, scored every epoch on fixed validation subsets, and
the epoch with the best mean validation identity is kept and scored once on the test
subsets.

Precision is fp32 throughout the head and the transformer. Half precision is not a
harmless speed-up here: the CRF loss normalises over every path through the lattice,
and under bf16 the same head loses several points of identity.

Run on one GPU::

    brieno train basecall -c configs/basecall_dna_r9.json -m tokenizer.pt -t transformer.pt
"""
import datetime
import json
import math
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from brieno.downstream.accuracy import METRIC, ChunkScores
from brieno.downstream.basecall import BasecallHeads, backend_upsample, pad_targets
from brieno.downstream.basecall_data import (AlignedCorpus, BasecallCollator, BlendedChunks,
                                             fixed_subset, open_chunks)
from brieno.downstream.features import (FrozenFeatures, context_input, load_transformer,
                                        transformer_hidden)
from brieno.downstream.heads import RawEncoder
from brieno.train.utils import (JsonlLogger, load_checkpoint, restore_rng_state, rng_state,
                                worker_init_fn)

__all__ = ["ALL_HEADS", "DECODERS", "make_features", "evaluate", "finetune_basecaller"]

ALL_HEADS = ["A", "B1", "B2", "C1", "C2", "D1"]
NEEDS_TRANSFORMER = ("C1", "C2")
DECODERS = ("auto", "koi", "fcd", "viterbi")

# Beam pruning thresholds. The two beam searches define theirs on different scales, so
# each keeps the value it was validated with rather than sharing one config entry.
KOI_BEAM_CUT = 100.0
FCD_BEAM_CUT = 1e-3


def resolve_decoder(name):
    """Pick the CRF decoder: ``auto`` takes the best one this installation has."""
    if name not in DECODERS:
        raise ValueError(f"decoder must be one of {DECODERS}, got {name!r}")
    if name != "auto":
        return name
    try:
        import koi.decode                                   # noqa: F401
        return "koi"
    except ImportError:
        pass
    try:
        import fast_ctc_decode                              # noqa: F401
        return "fcd"
    except ImportError:
        return "viterbi"


def make_features(x, feat, tf, heads, raw=None):
    """Raw chunks -> the input each head wants.

    C1 is computed without gradients; C2 lets them flow into the transformer, which is
    the whole difference between the two.
    """
    f = {}
    if feat is not None:
        lat, q, idx = feat(x)
        f.update({"A": lat, "B1": q, "B2": idx})
    if raw is not None:
        f["D1"] = raw(x)
    if "C1" in heads or "C2" in heads:
        ctx = context_input(tf, f["A"], f["B2"])
    if "C1" in heads:
        with torch.no_grad():
            f["C1"] = transformer_hidden(tf, ctx)
    if "C2" in heads:
        f["C2"] = transformer_hidden(tf, ctx)
    return f


def _loss(scores, targets_flat, target_lengths, seqdist):
    return seqdist.ctc_loss(scores, pad_targets(targets_flat, target_lengths), target_lengths)


def _decode(scores, seqdist, decoder, beam_width, blank_score, threads):
    from brieno.models import crf

    if decoder == "koi":
        seqs, _, _ = crf.crf_beam_decode(scores, beam_width=beam_width, beam_cut=KOI_BEAM_CUT,
                                         blank_score=blank_score)
        return seqs
    if decoder == "fcd":
        return crf.crf_fcd_beam_decode(seqdist, scores, beam_size=beam_width,
                                       beam_cut=FCD_BEAM_CUT, threads=threads)
    return crf.crf_viterbi_decode(seqdist, scores)


def evaluate(loader, feat, tf, raw, heads, model, seqdist, decode, device, max_batches=None):
    """Loss and chunk-level identity of every head over one evaluation set.

    ``decode`` is a callable ``scores -> list of strings``. Everything is put in eval
    mode for the pass and returned to how it was found.
    """
    model.eval()
    tf_was_training = tf is not None and tf.training
    if tf is not None:
        tf.eval()
    if raw is not None:
        raw.eval()

    scores = {h: ChunkScores() for h in heads}
    losses = {h: 0.0 for h in heads}
    n_batches = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            x = batch["inputs"].to(device, non_blocking=True)
            targets = batch["targets"].to(device)
            lengths = batch["target_lengths"].to(device)
            out = model(make_features(x, feat, tf, heads, raw))
            for h in heads:
                losses[h] += _loss(out[h], targets, lengths, seqdist).item()
                scores[h].add(decode(out[h]), batch["ref_strs"])
            n_batches += 1

    if raw is not None:
        raw.train()
    if tf_was_training:
        tf.train()
    model.train()
    return {h: dict(loss=losses[h] / max(n_batches, 1), **scores[h].summary()) for h in heads}


def _short(name):
    """``Lambda_Native_R9_x`` -> ``lambda``; ``<dataset>/<run>`` -> ``<run>``. Log lines only."""
    return name.split("/")[-1] if "/" in name else name.split("_")[0].lower()


def finetune_basecaller(config_path, tokenizer_ckpt, transformer_ckpt=None, heads=("C2",),
                        run_name=None, resume=None, seed=None, overrides=None,
                        decoder="auto", transformer_random=False, smoke=0):
    """Train one or more basecalling heads from a JSON config. Returns the output directory.

    ``smoke`` > 0 runs that many training batches per epoch and that many batches of
    every evaluation: a check that the installation and the data are in order, not a
    result.
    """
    heads = list(heads)
    unknown = [h for h in heads if h not in ALL_HEADS]
    if unknown:
        raise ValueError(f"unknown head(s) {unknown}; choose from {ALL_HEADS}")
    if "C1" in heads and "C2" in heads:
        raise ValueError("C1 and C2 cannot share a job: C2 fine-tunes the transformer "
                         "that C1 must see frozen. Run them separately.")
    use_tf = any(h in NEEDS_TRANSFORMER for h in heads)
    if use_tf and transformer_ckpt is None:
        raise ValueError(f"heads {NEEDS_TRANSFORMER} need --transformer")
    if transformer_random and heads not in (["C1"], ["C2"]):
        raise ValueError("--transformer-random is a control for one head at a time: "
                         "use it with --heads C1 or --heads C2")
    if not torch.cuda.is_available():
        raise RuntimeError("the CRF loss runs on CUDA only; no GPU is visible")
    from brieno.models import crf
    if crf.BACKEND is None:
        raise ImportError("the CRF loss needs a CUDA backend: install `ont-koi` (x86_64) "
                          "or `cupy` (any CUDA platform)")
    decoder = resolve_decoder(decoder)

    with open(config_path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    for section, key, value in (overrides or []):
        if value is not None:
            cfg[section][key] = value
    data, head_cfg, train = cfg["datasets"], cfg["head"], cfg["training"]

    seed = int(train.get("seed", 42)) if seed is None else int(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda:0")
    t0 = time.time()

    def log(*msg):
        print(f"[{time.time() - t0:7.0f}s]", *msg, flush=True)

    if resume:
        # A resumed run continues in the directory it was started in: the best
        # checkpoint so far and the history live there, and a fresh directory would
        # start without either.
        out_dir = os.path.dirname(os.path.abspath(resume))
    else:
        name = run_name or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(train["save_dir"], name)
    os.makedirs(out_dir, exist_ok=True)
    history = JsonlLogger(os.path.join(out_dir, "history.jsonl"))
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)
    log(f"writing to {out_dir}")
    log(f"heads {heads} | CRF backend {crf.BACKEND} | decoder {decoder} | metric {METRIC}")

    # ---------------------------------------------------------------- data
    root = data["root_dir"]
    max_ref_len = int(data.get("max_ref_len", 2047))
    batch = int(train["batch_size"])
    workers = int(train.get("num_workers", 8))
    subset_seed = int(data.get("subset_seed", 42))

    train_sets, weights = [], []
    for entry in data["train"]:
        ds = AlignedCorpus(os.path.join(root, entry["name"]), max_ref_len=max_ref_len)
        train_sets.append(ds)
        weights.append(float(entry["weight"]))
        log(f"train {entry['name']}: {len(ds):,d} chunks (weight {entry['weight']})")
    # A whole number of batches, so every epoch is the same number of optimizer steps.
    n_samples = max(1, int(train["epoch_chunks"]) // batch) * batch
    train_ds = BlendedChunks(train_sets, weights, n_samples)

    collate = BasecallCollator()
    val_names = list(data["validation"])
    val_loaders = {}
    for v in val_names:
        vd = fixed_subset(open_chunks(os.path.join(root, v), max_ref_len=max_ref_len),
                          int(data.get("validation_chunks", 2000)), subset_seed)
        val_loaders[v] = DataLoader(vd, batch_size=int(train.get("eval_batch_size", 64)),
                                    shuffle=False, num_workers=workers, pin_memory=True,
                                    collate_fn=collate)
        log(f"val {v}: {len(vd):,d} chunks (seed {subset_seed})")
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=workers,
                              pin_memory=True, drop_last=True, collate_fn=collate,
                              worker_init_fn=worker_init_fn, persistent_workers=workers > 0)

    # ---------------------------------------------------------------- front-end
    # A head trained from scratch (D1) only needs the tokenizer's shape, so the
    # tokenizer itself is built only when a head reads from it.
    use_tok = any(h != "D1" for h in heads)
    if use_tok:
        from brieno.data.encode import load_tokenizer
        tok, tok_cfg = load_tokenizer(tokenizer_ckpt, device=device)
        feat = FrozenFeatures(tok)
    else:
        tok_cfg = load_checkpoint(tokenizer_ckpt)["model_config"]
        feat = None
    hidden_dim = int(tok_cfg["hidden_dim"])
    downsample = int(tok_cfg.get("downsample", 16))
    upsample = backend_upsample(downsample)

    raw = RawEncoder(hidden_dim, downsample, dropout=0.0).to(device) if "D1" in heads else None

    tf = tf_cfg = None
    tf_train = "C2" in heads
    if use_tf:
        tf, tf_cfg = load_transformer(transformer_ckpt, device, trainable=tf_train,
                                      random_init=transformer_random)
        n_frames = 16384 // downsample
        if int(tf_cfg["max_seq_len"]) < n_frames:
            raise ValueError(f"the transformer's context ({tf_cfg['max_seq_len']}) is shorter "
                             f"than one chunk ({n_frames} frames)")

    seqdist = crf.CTC_CRF(state_len=int(head_cfg.get("state_len", 5)))
    backend_kw = dict(d=int(head_cfg["d"]), lstm_hidden=int(head_cfg["lstm_hidden"]),
                      n_lstm=int(head_cfg["n_lstm"]), dropout=float(head_cfg["dropout"]),
                      state_len=int(head_cfg.get("state_len", 5)),
                      crf_scale=float(head_cfg.get("crf_scale", 5.0)),
                      blank_score=float(head_cfg.get("blank_score", 2.0)), upsample=upsample)
    model = BasecallHeads(heads, hidden_dim, int(tok_cfg["num_quantizers"]),
                          int(tok_cfg["num_tokens"]),
                          ctx_dim=int(tf_cfg["d_model"]) if tf_cfg else None,
                          **backend_kw).to(device)

    log(f"tokenizer: downsample {downsample}x, {int(tok_cfg['num_quantizers'])} x "
        f"{int(tok_cfg['num_tokens'])} codes -> back-end upsample {upsample}x "
        f"({16384 // downsample * upsample} output frames per chunk)")
    for h in heads:
        log(f"  head {h}: {model.n_params(h) / 1e6:.2f} M parameters")
    if use_tf:
        log(f"transformer: {tf_cfg['n_layer']} layers, d_model {tf_cfg['d_model']}, "
            f"reads {'code ids' if tf.input_mode == 'token' else 'the continuous latent'}, "
            + (f"fine-tuned at lr {float(train['transformer_lr']):g}" if tf_train else "frozen")
            + ("  (randomly initialised control)" if transformer_random else ""))

    beam_width = int(cfg.get("decode", {}).get("beam_width", 32))
    blank_score = backend_kw["blank_score"]

    def decode(scores):
        return _decode(scores, seqdist, decoder, beam_width, blank_score, max(1, workers))

    # ---------------------------------------------------------------- optimisation
    lr = float(train["lr"])
    tf_params = [p for p in tf.parameters() if p.requires_grad] if tf_train else []
    groups = [{"params": list(model.parameters())
               + (list(raw.parameters()) if raw is not None else []), "lr": lr}]
    if tf_train:
        # The transformer is pretrained and the head is not: a head-sized step would
        # undo what pretraining established.
        groups.append({"params": tf_params, "lr": float(train["transformer_lr"])})
    opt = torch.optim.AdamW(groups, lr=lr, weight_decay=float(train["weight_decay"]),
                            betas=tuple(train.get("betas", (0.9, 0.98))),
                            eps=float(train.get("eps", 1e-6)))

    epochs = int(train["epochs"])
    steps_per_epoch = smoke if smoke else len(train_loader)
    total = epochs * steps_per_epoch
    warmup = int(train.get("warmup_steps", 1000))
    # Linear warm-up multiplied by a cosine over the whole run. The cosine starts at
    # step 0, not at the end of the warm-up; the released models were trained this way.
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, warmup))
        * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, total)))))
    clip = float(train.get("grad_clip", 2.0))

    def head_params(h):
        # A head is clipped on its own parameters, so a blown-up gradient in one head
        # cannot scale the update of another through a shared norm.
        return (model.parameters_of(h)
                + (list(raw.parameters()) if h == "D1" else [])
                + (tf_params if h == "C2" else []))

    start_epoch = 0
    best = {h: (-1.0, None) for h in heads}
    if resume:
        state = load_checkpoint(resume, map_location=device)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        if raw is not None:
            raw.load_state_dict(state["raw"])
        if tf_train:
            tf.load_state_dict(state["tf"])
        start_epoch = state["epoch"]
        best = {h: tuple(v) for h, v in state["best"].items()}
        restore_rng_state(state["rng"])
        log(f"resumed from {resume}: epoch {start_epoch} done")

    def head_checkpoint(h, epoch, score, val):
        return {"format": 1, "head_name": h, "head": model.head_state(h), "head_config": backend_kw,
                "epoch": epoch, "val_nw": score, "val": val,
                "metric": METRIC, "decoder": decoder,
                "tokenizer": os.path.basename(tokenizer_ckpt),
                "transformer": os.path.basename(transformer_ckpt) if use_tf else None,
                **({"transformer_state": tf.state_dict()} if h == "C2" else {}),
                **({"raw_encoder": raw.state_dict()} if h == "D1" else {})}

    # ---------------------------------------------------------------- loop
    n_bad = 0
    for epoch in range(start_epoch, epochs):
        model.train()
        if tf_train:
            tf.train()
        t_epoch = time.time()
        running = {h: 0.0 for h in heads}
        step = 0

        for batch_ in train_loader:
            if smoke and step >= smoke:
                break
            x = batch_["inputs"].to(device, non_blocking=True)
            targets = batch_["targets"].to(device)
            lengths = batch_["target_lengths"].to(device)

            out = model(make_features(x, feat, tf, heads, raw))
            losses = {h: _loss(out[h], targets, lengths, seqdist) for h in heads}

            # A non-finite loss in one head would poison every other head through the
            # summed backward pass, so the batch is dropped rather than the run.
            bad = [h for h in heads if not torch.isfinite(losses[h])]
            if bad:
                n_bad += 1
                step += 1
                log(f"epoch {epoch + 1} step {step}: non-finite loss in {bad}, batch skipped "
                    f"({n_bad} so far)")
                if n_bad > 50:
                    raise RuntimeError("too many non-finite losses")
                continue

            opt.zero_grad(set_to_none=True)
            sum(losses.values()).backward()
            for h in heads:
                if not torch.isfinite(torch.nn.utils.clip_grad_norm_(head_params(h), clip)):
                    bad.append(h)
            if bad:
                n_bad += 1
                step += 1
                opt.zero_grad(set_to_none=True)
                log(f"epoch {epoch + 1} step {step}: non-finite gradient in {bad}, step skipped "
                    f"({n_bad} so far)")
                if n_bad > 50:
                    raise RuntimeError("too many non-finite gradients")
                continue

            opt.step()
            sched.step()
            for h in heads:
                running[h] += losses[h].item()
            step += 1
            if step % 200 == 0:
                log(f"epoch {epoch + 1} step {step}/{steps_per_epoch}  loss "
                    + " ".join(f"{h}={running[h] / step:.3f}" for h in heads)
                    + f"  lr {sched.get_last_lr()[0]:.2e}"
                    + f"  ({step * batch / (time.time() - t_epoch):.0f} chunks/s)")

        val = {v: evaluate(val_loaders[v], feat, tf, raw, heads, model, seqdist, decode, device,
                           max_batches=smoke or None) for v in val_names}
        val_mean = {h: float(np.mean([val[v][h]["nw_mean"] for v in val_names])) for h in heads}
        train_loss = {h: running[h] / max(step, 1) for h in heads}
        history.log(epoch=epoch + 1, seconds=time.time() - t_epoch, train_loss=train_loss,
                    val_nw_mean=val_mean, val=val)
        log(f"epoch {epoch + 1}/{epochs} ({time.time() - t_epoch:.0f}s): " + " | ".join(
            f"{h} loss {train_loss[h]:.3f} val {val_mean[h] * 100:.2f} ["
            + " ".join(f"{_short(v)} {val[v][h]['nw_mean'] * 100:.2f}" for v in val_names) + "]"
            for h in heads))

        # One checkpoint per head: the epoch with the best mean identity over all
        # validation sets, so that no single set decides which model is kept.
        for h in heads:
            if val_mean[h] > best[h][0]:
                best[h] = (val_mean[h], epoch + 1)
                torch.save(head_checkpoint(h, epoch + 1, val_mean[h],
                                           {v: val[v][h] for v in val_names}),
                           os.path.join(out_dir, f"best_{h}.pt"))

        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": epoch + 1,
                    "best": {h: list(v) for h, v in best.items()}, "rng": rng_state(),
                    **({"raw": raw.state_dict()} if raw is not None else {}),
                    **({"tf": tf.state_dict()} if tf_train else {})},
                   os.path.join(out_dir, "last.pt"))

    # ---------------------------------------------------------------- test
    for h in heads:
        state = load_checkpoint(os.path.join(out_dir, f"best_{h}.pt"), map_location=device)
        model.load_head_state(h, state["head"])
        if h == "C2":
            tf.load_state_dict(state["transformer_state"])
        if h == "D1":
            raw.load_state_dict(state["raw_encoder"])

    results = {h: {} for h in heads}
    for t in data.get("test", []):
        td = fixed_subset(open_chunks(os.path.join(root, t), max_ref_len=max_ref_len),
                          int(data.get("test_chunks", 20000)), subset_seed)
        loader = DataLoader(td, batch_size=int(train.get("eval_batch_size", 64)),
                            num_workers=workers, pin_memory=True, collate_fn=collate)
        log(f"test {t}: {len(td):,d} chunks ...")
        scored = evaluate(loader, feat, tf, raw, heads, model, seqdist, decode, device,
                          max_batches=smoke or None)
        for h in heads:
            results[h][t] = scored[h]
            m = scored[h]
            log(f"  {h}: identity mean {m['nw_mean'] * 100:.2f} median {m['nw_median'] * 100:.2f}"
                f" | mis/ins/del {m['mismatch_rate'] * 100:.2f}/{m['ins_rate'] * 100:.2f}/"
                f"{m['del_rate'] * 100:.2f} | epoch {best[h][1]}")

    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump({"heads": heads, "metric": METRIC, "decoder": decoder,
                   "crf_backend": crf.BACKEND,
                   "tokenizer": os.path.basename(tokenizer_ckpt),
                   "transformer": os.path.basename(transformer_ckpt) if use_tf else None,
                   "best_epoch": {h: best[h][1] for h in heads},
                   "best_val_nw": {h: best[h][0] for h in heads},
                   "test": results, "config": cfg, "seconds": time.time() - t0}, fh, indent=2)
    log("done")
    return out_dir
