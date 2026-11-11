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
      ├── brieno train classify ──▶  demultiplexing / species heads
      │
      └── brieno train basecall ──▶  basecalling head
```

The two classification tasks share one command; basecalling has its own, because its
head emits a sequence rather than a label. The modification head is fine-tuned by a
command that is being prepared.

The transformer can also be pretrained on the tokenizer's continuous latent instead of
its tokens, which drops the `encode` step; see
[Reading the latent instead of the tokens](#reading-the-latent-instead-of-the-tokens-experimental).

---

## Install

```bash
git clone https://github.com/guangzhaocs/Brieno.git
cd Brieno
pip install -e ".[fast5,pod5]"
```

`fast5` and `pod5` are separate extras because most people have data in one format
only. Install just the one you need, or both.

Training needs a CUDA GPU. Basecalling additionally needs a CUDA backend for the CRF
loss and benefits from two more packages; see
[step 7](#7-fine-tune-a-basecalling-head). None of that is required for the tokenizer,
the transformer or the classification heads.

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
Biases are excluded from weight decay; every other parameter is decayed, LayerNorm
gains included. The count of each group is printed at startup.

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

### Reading the latent instead of the tokens (experimental)

Everything above describes a transformer that reads token ids, which is what the
released model is. `model.input_mode` selects what the transformer reads:

| `input_mode` | the transformer reads | it predicts |
|---|---|---|
| `"token"` (default) | the code ids of each frame, one embedding table per codebook | the code ids of the hidden frames |
| `"latent"` | the tokenizer's encoder output for each frame, before quantization | the code ids of the hidden frames |

The task is the same in both — the same frames hidden, the same targets, the same
`layer_weights`. What moves is the quantizer: in token mode it sits in the input path,
so the transformer only ever sees what the codebooks kept; in latent mode it sits in
the label path only, the transformer sees the full-precision latent, and at inference
the quantizer is not needed at all.

```bash
brieno train transformer -c configs/transformer_latent_dna_r9.json --tokenizer tokenizer.pt
```

What changes in the workflow:

- **No `encode` step.** `datasets.root_dir` points at the *signal* corpus from step 2,
  not at a tokenized one. A stored latent would be about a hundred times the size of
  the signal it came from, so the frozen tokenizer runs during training instead and
  produces both the input and the targets for each batch. Its encoder is small next
  to the transformer, so this adds little to the cost of a step.
- **A tokenizer is required**, as `--tokenizer` or a top-level `"tokenizer"` entry in
  the config. `vocab_size`, `num_vq_layers` and `latent_dim` are properties of that
  tokenizer and are filled in from it; a config that contradicts it is refused. The
  targets are computed in full precision, so they are the ids `brieno train encode`
  would write for the same chunks.
- **Hidden frames** are replaced by a learned mask vector (80 %), by the latent of
  another frame in the batch (10 %), or left as they are (10 %) — the same split the
  token mode applies to ids.

The checkpoint records its input mode, so nothing downstream needs a flag:
`brieno train classify` feeds heads `C1` and `C2` token ids or the latent according to
the transformer it is given.

Two cautions. No released weights were trained this way, and the recipe in
`configs/transformer_latent_dna_r9.json` is the token recipe with the input switched,
not a tuned one. And the pretraining loss of the two modes cannot be compared: they
solve the same task from different inputs, so a lower loss in one says nothing about
which transformer is the better starting point. Compare them on a downstream task.

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

`C1` and `C2` feed the transformer whatever it was pretrained on: token ids for the
released model, the continuous latent for one pretrained in
[latent mode](#reading-the-latent-instead-of-the-tokens-experimental). The checkpoint
says which, so the command line is the same.

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

`--resume <last.pt>` continues an interrupted run, in the directory the checkpoint is
in. The checkpoint carries the random state as well as the weights, so a resumed run draws the epoch it is actually on
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

---

## 7. Fine-tune a basecalling head

A basecalling head turns a chunk of signal into the bases it contains. It is trained
on chunks paired with the stretch of reference they align to, and it ends in the same
CTC-CRF output layer, loss and beam search as Dorado and Bonito, so that a difference
in accuracy comes from what is underneath the output layer and not from how sequences
are scored.

```bash
pip install -e ".[decode,align,cuda]"     # see "What to install" below

brieno train basecall -c configs/basecall_dna_r9.json \
    -m dna_r9.4.1_tokenizer@v1.0/tokenizer.pt \
    -t dna_r9.4.1_token_tf_pretrained@v1.0/transformer.pt \
    --heads C2 -n my_run
```

`--smoke 5` runs five batches of everything and exits. Use it first: it tells you in a
minute whether the data, the weights and the CRF backend are in order, which a real
run tells you only after it has loaded the whole corpus.

### The data

```
<root>/<dataset>/<run>/manifest.json
                       chunks_0000.signal.npy       (N, 1, 16384) float16
                       chunks_0000.refs_label.npy   (N, max_len) int8   0 = pad, 1..4 = ACGT
                       chunks_0000.refs_len.npy     (N,) int32
```

`manifest.json` lists the shards under `"files"`; each entry gives `signal_file`,
`refs_label_file`, `refs_len_file` and `n_chunks`. A dataset is a directory of runs,
and the config names datasets (or single runs, as `<dataset>/<run>`) relative to
`datasets.root_dir`.

The signal is normalised and chunked exactly as for pretraining. The label of a chunk
is the piece of **reference** its signal aligns to — obtained by aligning each read to
its genome at signal level and cutting the alignment at the chunk boundaries — and not
the output of another basecaller. Training on a basecaller's calls teaches its errors
along with its skill, and caps the student at the teacher.

`max_ref_len` drops chunks with more bases than the head can emit. The CRF needs at
least one output frame per base; at 4096 frames per chunk the limit of 2047 leaves two.

As with the classification tasks, hold out whole **runs**. The released recipe
validates and tests on runs — and, for the human sets, on a different individual —
that appear nowhere in training. Its validation and test subsets are drawn from the
same held-out runs, so the epoch that is kept was chosen with those runs in view; the
test numbers are a fair comparison between heads trained this way, and a slightly
generous estimate for a run the model has never been selected on.

### The head

Every head is the same back-end on a different representation, and the letters mean
what they mean in [step 6](#choosing-a-head):

```
features [B, L, D]
   │  Linear(D → d)
   │  transposed convolutions up to 4 samples per output frame
   │  bidirectional LSTM, n_lstm layers × lstm_hidden
   ▼
CRF output layer   Linear → 4^(state_len + 1) scores per frame, tanh × crf_scale
```

The output rate is fixed at one frame per four samples whatever the tokenizer's
stride, so a 16384-sample chunk always gives 4096 frames: the 8× tokenizer is
upsampled 2×, a 16× one 4×. Dorado's R9.4.1 models use five samples per frame.

The **`head`** section of the config holds the configuration that was benchmarked
against other basecallers, and these are also the defaults in the code:

| key | value | |
|---|---|---|
| `d` | 512 | width after the projection and through the upsampling |
| `lstm_hidden` | 512 | per direction |
| `n_lstm` | 5 | |
| `dropout` | 0.0 | |
| `state_len` | 5 | CRF state = the last five bases, as in Dorado's sup models |
| `crf_scale`, `blank_score` | 5.0, 2.0 | as in Dorado |

That back-end has 35 M parameters. It is deliberately larger than a head needs to be
for comparing representations: widening it from 256 to 512 raised chunk identity by
about 0.8 points on each representation it was tried on, so the head worth shipping
is not the smallest one that ranks the inputs correctly. `n_lstm: 0` removes the
sequence model altogether, leaving the head no context of its own.

### The recipe

**`datasets`** — `train` is a weighted mixture, sampled with replacement, so the mix
is what you set and not what the corpus sizes happen to be. `validation` and `test`
are fixed subsets (`validation_chunks`, `test_chunks`, drawn once with `subset_seed`):
they must be the same chunks at every epoch, in every run and on every machine, or
their scores stop being comparable.

**`training`** — an epoch is `epoch_chunks` chunks, so its length does not depend on
the size of the corpus. `lr` is the head's learning rate and `transformer_lr` the
transformer's when `C2` fine-tunes it, ten times smaller for the same reason as in
step 6. The schedule is a linear warm-up multiplied by a cosine over the whole run.
Gradients are clipped per head (`grad_clip`), so one head's bad batch cannot scale
another's update when several share a job.

The head and the transformer run in **fp32**. This is not caution: the CRF loss
normalises over every path through the lattice, and under bf16 the same head loses
more than two points of identity. Only the frozen tokenizer encoder uses bf16, where
the GPU supports it.

The checkpoint kept for each head is the epoch with the best **mean** identity over
all validation sets. Selecting on one set picks the epoch that happens to suit that
set; the mean is harder to flatter.

`--resume <last.pt>` continues an interrupted run in the directory the checkpoint is
in, random state included, so the data stream carries on from where it stopped
instead of replaying the first epochs.

### What to install

| extra | package | what it is for |
|---|---|---|
| `cuda` | `cupy` | the CRF loss, on any CUDA platform |
| — | `ont-koi` | the same kernels and Dorado's GPU beam search; x86_64 wheels only, install it yourself |
| `decode` | `fast-ctc-decode` | CRF beam search on the CPU, everywhere |
| `align` | `parasail` | the identity metric; without it `edlib` is used, with unit costs |

One of `ont-koi` and `cupy` is required. The rest decide *how* results are scored,
and the choice is recorded in `metrics.json` because it changes the numbers:

- **Decoder** (`--decoder`): `koi` is Dorado's beam search, `fcd` the open-source one,
  `viterbi` the best path without a beam. `auto` takes the first of those that is
  installed. At beam width 32 the two beam searches agree on 99.7 % of bases and
  `fcd` scores about 0.09 points lower. Compare numbers only within one decoder.
- **Metric**: identity of a global alignment scored match 2, mismatch −1, gap open 8,
  gap extend 4 — the chunk-level metric of Pagès-Gallego & de Ridder (2023). The
  `edlib` fallback uses unit costs and does not give the same values.

### Output

```
<save_dir>/<run name>/
    config.json         the config this run actually used
    history.jsonl       per epoch: training loss, and every validation set's scores
    best_<head>.pt      the epoch with the best mean validation identity
    last.pt             for --resume
    metrics.json        test-set scores of the kept epoch, with decoder and metric
```

`best_<head>.pt` carries the head's configuration, and for `C2` the fine-tuned
transformer, so it can be rebuilt without the config file.

What to watch:

- **`nw_mean`** — mean per-chunk identity, `matches / (matches + mismatches +
  insertions + deletions)`. This is what epochs are selected on. `nw_median` sits
  about a point above it; the gap is the tail of chunks that go badly.
- **`mismatch_rate` / `ins_rate` / `del_rate`** — which kind of error is left.
- **`len_ratio`** — called length over reference length. Well below 1 means the head
  is dropping bases, well above means it is stuttering.

These are chunk-level numbers, measured on 16384-sample pieces against their own
reference. They are the right thing to select epochs on and to compare heads with,
but they are not read accuracy: basecalling whole reads, with chunks overlapped and
stitched, is the job of the `brieno basecaller` command, which is being prepared.

