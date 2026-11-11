# Training guide

How to build a Brieno model from raw nanopore reads.

If you only want to *run* a trained model — basecalling, demultiplexing, species
identification, modification calling — you do not need any of this. See the
[README](../README.md).

Every command below has its own `--help` with the full list of flags and their
defaults; this guide covers the workflow and the ideas behind it, not the flag table.

---

## The pipeline

```
raw FAST5/POD5
      │
      │  brieno train prep
      ▼
  signal chunks (.npy shards)
      │
      │  brieno train merge
      ▼
  one .npy per dataset
      │
      │  brieno train tokenizer
      ▼
  signal tokenizer
      │
      │  brieno train encode
      ▼
  tokenized corpus
      │
      │  brieno train transformer
      ▼
  token transformer
      │
      │  brieno train classify
      ▼
  demultiplexing / species heads
```

Basecalling and modification heads are fine-tuned by commands of their own, which are
being prepared; the two classification tasks share one command and are here.

---

## Install

```bash
git clone https://github.com/guangzhaocs/Brieno.git
cd Brieno
pip install -e ".[fast5,pod5]"
```

`fast5` and `pod5` are separate extras because most people have data in one format
only. Install just the one you need, or both.

Training needs a CUDA GPU. The CRF parts of the downstream tasks additionally need
`ont-koi` (x86_64) or `cupy` (any CUDA platform) — not required for the tokenizer.

---

## 1. Chunk the raw signal

```bash
brieno train prep -i /data/run1/pod5 -o /data/chunks/Ecoli_Native_R9 -j 8
```

Per read, this does three things:

1. **Normalise** the whole read by median / MAD. Nanopore current is measured in
   picoamperes and drifts between pores and runs; the median/MAD transform puts every
   read on the same scale without assuming a particular current level.
2. **Trim** the open-pore and stall region at the read start, using bonito's window
   scan. The threshold depends on the chemistry, so set `--modality` to match your
   data (`dna_r9`, `dna_r10`, `rna_002`, `rna_004`).
3. **Cut** the rest into overlapping fixed-length windows, written as `float16` shards
   of shape `(N, 1, chunk_size)`.

Output is one or more `<input stem>_partN.npy` files, at most
`--max-chunks-per-file` chunks each.

Useful flags: `-j/--workers` processes whole files in parallel (they are independent,
so this scales linearly with cores until the filesystem saturates); `--resume` skips
inputs that already have output; `-r/--recursive` walks subdirectories.

### Why this is not what the task commands use

`prep` is for **training data**. The task commands (`basecaller`, `demux`, …) chunk
reads in memory with a *different* cut and never read these files. Two reasons they
must differ:

- `prep` **trims** the read start; inference must not, because the trimmed samples
  carry bases that belong in the output sequence.
- `prep` **drops** reads shorter than one chunk and discards the leading remainder;
  inference pads short reads and covers every sample, because a read that produced no
  output at all is a read missing from your results.

Do not feed `prep` output to a task command, and do not change `prep` to match one.

---

## 2. Merge the shards into one file per dataset

The tokenizer trainer reads one array per dataset, not per input file:

```
<pretrain_root>/
    Ecoli_Native_R9_PRJEB51164.npy        (N, 1, 16384) float16
    Hsapiens_Native_R9_HG001Prometh.npy
    ...
```

because the sampler needs a single index space per dataset to draw from, and a stable
train/validation split across runs.

```bash
brieno train merge -i /data/chunks/Ecoli_Native_R9 \
                   -o /data/corpus/Ecoli_Native_R9_PRJEB51164.npy
```

The merge is two passes over memory-mapped files — measure, then fill — so the corpus
never has to fit in RAM. All shards must agree on dtype and chunk shape; a mismatch
fails before anything is written and names the offending file. An existing output is
not overwritten unless you pass `--force`.

**The output filename is the dataset name**, and that name is the key you will use in
the training config. Name it after the dataset, not after the run it came from.

For a whole corpus, loop in the shell rather than submitting one job per dataset:

```bash
for d in /data/chunks/*/; do
    name=$(basename "$d")
    brieno train merge -i "$d" -o "/data/corpus/${name}.npy"
done
```

---

## 3. Pretrain the tokenizer

The tokenizer learns to reconstruct raw signal through a residual vector quantizer.
The loss is `L1(reconstruction, signal) + vq_loss` — no labels of any kind, so any
corpus of raw reads will do.

```bash
# one GPU
brieno train tokenizer -c configs/tokenizer_dna_r9.json

# four GPUs on one node
torchrun --nproc_per_node=4 -m brieno.cli train tokenizer -c configs/tokenizer_dna_r9.json
```

### The config

Start from [`configs/tokenizer_dna_r9.json`](../configs/tokenizer_dna_r9.json), which
is the recipe of the released R9.4.1 tokenizer. Three sections:

**`datasets`** — where the corpus is and how to mix it.

`pretrain_weights` is a list of `{name, weight}`. Weights are normalised, so they are
proportions of the training stream, *not* of the data on disk: a dataset with a
weight of 0.12 contributes 12 % of the chunks the model sees however large its file
is. This is what keeps a corpus balanced when one organism happens to have ten times
more reads than another.

`validation` is a separate list of dataset names. Each file is split once by a fixed
seed into a training part and a validation part (at most 10 % of the file, capped by
`validation_size_per_dataset`), so the held-out chunks are the same on every run and
on every machine. Listing a dataset in `validation` but not in `pretrain_weights` is
deliberate and useful: it measures generalisation to an organism the model never
trained on.

**`training`** — `epoch_length` is the number of *draws* per epoch, not a pass over
the data: each draw picks a dataset by weight and then a random chunk inside it. This
makes epochs comparable between runs with different corpus sizes. `save_dir` is the
parent; each run gets a subdirectory (`--run-name`, or a timestamp).

**`model`** — `downsample` sets the token rate (8 means 16,384 samples become 2,048
frames), `num_quantizers` × `num_tokens` is the codebook stack, and `quant_dim`
quantises in a projected low-dimensional space rather than the full hidden width.
`commitment_cost` and `ema_decay` control the quantizer; the codebooks are EMA-tracked,
so the optimizer only ever sees the encoder and decoder.

Anything in the config can be overridden on the command line (`--lr`, `--epochs`,
`--commitment-cost`, …), which is convenient for sweeps without writing a config per
run.

### Output

```
<save_dir>/<run name>/
    config.json          the config this run actually used, overrides applied
    history.jsonl        one JSON object per logged step and per epoch
    tokenizer_ep<N>.pt   checkpoint after each epoch
    tokenizer_best.pt    lowest validation loss so far
```

Checkpoints hold a plain tokenizer `state_dict` plus the optimizer, scheduler and
scaler state, so `--resume <checkpoint>` continues a run that hit a wall-clock limit.
On a cluster, write `save_dir` to shared storage rather than node-local disk, or a
timed-out job leaves you with nothing to resume from.

Metrics are written to `history.jsonl` rather than an experiment-tracking service, so
a run's history lives next to its weights and can be read without an account:

```bash
python -c "
import json
for line in open('history.jsonl'):
    r = json.loads(line)
    if r['split'] == 'val':
        print(r['epoch'], round(r['recon'], 4), round(r['vq'], 4), r['active_tokens'])
"
```

What to watch:

- **`recon`** — reconstruction L1. The headline number, but note that below roughly
  5 bits per sample it stops predicting how useful the tokenizer is downstream, so do
  not optimise it past the point where the downstream task stops improving.
- **`active_tokens` / `dead_rate`** — how much of each codebook is in use. A codebook
  collapsing to a handful of entries is the classic VQ failure, and it shows up here
  long before it shows up in the loss.
- **`perplexity`** — the effective number of codebook entries used, per layer; a
  smoother version of the same signal.

---

## 4. Encode the corpus into tokens

The transformer trains on token sequences, not on signal, so the corpus has to be run
through the finished tokenizer once:

```bash
brieno train encode -m runs/my-tokenizer/tokenizer_best.pt \
                    -i /data/corpus/Ecoli_Native_R9_PRJEB51164.npy \
                    -o /data/tokens/Ecoli_Native_R9_PRJEB51164.npy
```

`(N, 1, chunk_size)` float16 signal becomes `(N, num_quantizers, seq_len)` uint16 token
ids, written through a memory-mapped output so peak memory stays flat. The tokenizer
checkpoint carries its own `model_config`, so there is no config to pass and no way for
the architecture to disagree with the weights.

Keep the dataset filenames: they are the keys in the transformer config, exactly as
they were in the tokenizer config.

A tokenizer whose codebook k-means never completed is refused rather than used — its
codebook is still random, and encoding with it would produce meaningless tokens that
nothing downstream would flag.

Loop over the corpus the same way as for the merge step:

```bash
for f in /data/corpus/*.npy; do
    brieno train encode -m runs/my-tokenizer/tokenizer_best.pt \
                        -i "$f" -o "/data/tokens/$(basename "$f")"
done
```

---

## 5. Pretrain the transformer

Spans of token positions are hidden and the model predicts them from their context.
This is the stage that turns a per-chunk code into a contextual representation, and it
is what the downstream heads attach to.

```bash
# one GPU
brieno train transformer -c configs/transformer_dna_r9.json

# four GPUs on one node
torchrun --nproc_per_node=4 -m brieno.cli train transformer -c configs/transformer_dna_r9.json
```

### The config

Start from [`configs/transformer_dna_r9.json`](../configs/transformer_dna_r9.json), the
recipe of the released R9.4.1 transformer. `datasets` works as it does for the
tokenizer, except `root_dir` points at the tokenized corpus from step 4.

**`masking`** — `mask_ratio` is the fraction of positions hidden and `min_span` the
length of each hidden run. Spans rather than single positions matter here: neighbouring
signal frames are highly correlated, so a model asked to fill in one isolated position
can interpolate from its neighbours without learning anything. The start probability is
corrected for span overlap, so the realised fraction matches `mask_ratio` instead of
overshooting it by `min_span`.

The same positions are masked in **every** quantizer layer at once. Masking a position
in one layer only would leave the model the same information through another codebook,
and it would learn to copy rather than to predict.

`layer_weights` sets each quantizer layer's share of the loss. The released recipe uses
`[1.0, 0.5, 0.2, 0.1]`: the first codebook carries most of the signal, and later
residual layers are progressively closer to noise, so weighting them equally would
spend most of the model's capacity predicting what cannot be predicted.

**`training`** — `epoch_length` is optimizer steps per GPU per epoch, so an epoch means
the same amount of work however many GPUs are used. `grad_accum_steps` multiplies the
effective batch without the memory of a larger one; the effective global batch is
`batch_size × world_size × grad_accum_steps`, printed at startup so it is not left to
arithmetic.

`betas` and `eps` default to `(0.9, 0.98)` and `1e-6`, the transformer convention rather
than PyTorch's AdamW defaults, and the values the released weights were trained with.
Biases and LayerNorm gains are automatically excluded from weight decay.

### Output

```
<save_dir>/<run name>/
    config.json            the config this run actually used
    history.jsonl          per-step and per-epoch metrics
    transformer_ep<N>.pt   checkpoint after each epoch
    transformer_best.pt    lowest validation loss so far
```

`--resume <checkpoint>` continues a run that hit a wall-clock limit.

What to watch in `history.jsonl`:

- **`loss`** — masked-token cross-entropy, weighted across layers.
- **`top1_L1` … `top1_L4`** — per-layer accuracy at masked positions. L1 should climb
  well above the others; later layers staying near chance is expected, not a fault,
  and is why their loss weights are small.

---

## 6. Fine-tune a classification head

Barcode demultiplexing and species identification are the same problem — label a read
from its signal — so they are the same command. The number of classes comes from the
dataset, and nothing in the code knows which task it is running:

```bash
brieno train classify -r /data/barcodes --setting setting1 -f 1 \
    -m dna_r9.4.1_tokenizer@v1.0/tokenizer.pt \
    -t dna_r9.4.1_token_tf_pretrained@v1.0/transformer.pt \
    --heads C2 --task-name demux -o runs/
```

### The dataset

```
<root>/meta/<class>/<run>/chunks/chunks_part0000.npy   (N, 1, 16384) float16
                         chunks/chunk_index.npy        (N, 3) int32
<root>/<setting>/labels.json                           {"classes": [...]}
<root>/<setting>/fold<k>/{train,val,test}.tsv          one row per run
```

Splits are stored as **indices, never as copies of the signal**: a fold says which
chunks of which runs belong to it, so one set of chunk files serves every fold and a
new split costs a table rather than a dataset.

Hold out whole **runs**, not reads. Reads from one flow cell share its pore ageing,
its temperature and its particular batch of chemistry, so a model that has seen any
read of a run has seen most of what distinguishes that run. A within-run split
measures how well the model memorised the run; a leave-run-out split measures what
you actually want to know.

### Choosing a head

Each head sees a different representation and is otherwise identical — the same
`ChunkHead`, the same data, the same schedule. That is what makes the comparison
mean something.

| head | input | needs `--transformer` |
|---|---|---|
| `A` | the tokenizer's continuous encoder latent | no |
| `B1` | the quantized latent | no |
| `B2` | token ids alone, through a learned embedding | no |
| `C1` | token ids through the pretrained transformer, frozen | yes |
| `C2` | the same, with the transformer fine-tuned | yes |
| `D1` | the encoder architecture trained from scratch on the task | no |

`C2` is usually the most accurate and `C1` close behind, but `C1` has a property `C2`
does not: its transformer is unchanged, so every `C1` head in existence shares one
front-end and several tasks can run in a single pass over the reads. `D1` is the
control — same encoder shape, no tokenizer, no transformer, no pretraining — and is
what any claim about the value of pretraining has to be measured against.

Several heads can share one job, which computes the frozen front-end once for all of
them:

```bash
brieno train classify ... --heads A,B1,B2,C1
```

`C1` and `C2` cannot: `C2` fine-tunes the transformer that `C1` must see frozen, so
running them together would quietly contaminate the frozen head while every metric
still looked reasonable. The command refuses rather than letting that happen.

### Training

Classes in these datasets differ in size by an order of magnitude — a run yields as
many chunks as it has reads — so each epoch draws `--per-class` chunks from every
class rather than sampling the natural distribution. An epoch is therefore a fixed
amount of work, comparable between datasets.

`--lr` sets the head's learning rate and `--tf-lr` the transformer's when `C2` is
fine-tuning. They are deliberately different: the transformer arrives pretrained and
the head does not, and a head-sized step applied to the transformer undoes what
pretraining established. The schedule is one-cycle over the whole run.

`--resume <last.pt>` continues an interrupted run. The checkpoint carries the random
state as well as the weights, so a resumed run draws the epoch it is actually on
instead of replaying the first one — with a per-class sampler, replaying means
training on the same chunks again and again while the loss curve looks healthy.

### Output

```
<out-root>/<task-name>_<heads>/<run name>/
    history.jsonl        per-epoch loss and validation accuracy
    best_<head>.pt       the epoch with the best validation accuracy, per head
    last.pt              resume point
    metrics.json         test-set results
    per_read_<head>.tsv.gz   one row per read: true and predicted class
```

`metrics.json` reports the test split at chunk level and at read level, the latter
under several aggregations. Which one is right depends on the task: a native barcode
sits in the first chunk of a read, so `read_acc_first1_mean_logp` is the honest number
there, while a species call may use the whole read and `read_acc_mean_logp` is the one
to quote. All of them are reported so the choice stays visible rather than buried.

Per-class recall and the confusion matrix are included, as is per-run recall — a model
that is excellent on average and poor on one run is a model that has learned something
about the runs rather than the classes.
