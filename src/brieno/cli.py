"""The ``brieno`` command line.

Two layers, split by audience:

* **Using a trained model** — one top-level subcommand per task::

      brieno basecaller    raw signal -> basecalls
      brieno demux         raw signal -> one barcode per read
      brieno classifier    raw signal -> one species per read
      brieno modcaller     raw signal -> modified bases
      brieno download      fetch model weights

* **Building a model** — everything developer-facing lives under ``brieno train``::

      brieno train prep       raw FAST5/POD5 -> normalised signal chunks
      brieno train merge      chunk shards -> one array per dataset
      brieno train tokenizer  pretrain the signal tokenizer
      brieno train encode     signal chunks -> token ids
      brieno train transformer  pretrain the token transformer
      brieno train classify     fine-tune a classification head

Subcommands are registered here but their implementations are imported lazily, so
``brieno --help`` does not pay for importing torch.
"""
import argparse
import sys

from brieno import __version__

VERSION_TEXT = f"""brieno version {__version__}
Written by Guangzhao Cheng.

Copyright 2026 FIMM, University of Helsinki"""


class _VersionAction(argparse.Action):
    """argparse's own version action re-wraps the text; this one prints it verbatim."""

    def __init__(self, option_strings, dest=argparse.SUPPRESS,
                 default=argparse.SUPPRESS, help="show the version and exit"):
        super().__init__(option_strings=option_strings, dest=dest, default=default,
                         nargs=0, help=help)

    def __call__(self, parser, namespace, values, option_string=None):
        print(VERSION_TEXT)
        parser.exit()


def _print_help_and_fail(parser):
    """A namespace command called with no subcommand: show its help, exit non-zero."""
    def run(_args):
        parser.print_help()
        return 1
    return run


# --------------------------------------------------------------- brieno train prep
def _add_train_prep(subparsers):
    p = subparsers.add_parser(
        "prep",
        help="turn raw FAST5/POD5 into normalised signal chunks",
        description=(
            "Read raw signal, normalise each read by median/MAD, trim the open-pore and "
            "stall region at the read start, and cut the rest into overlapping fixed-length "
            "chunks, written as float16 .npy shards of shape (N, 1, chunk-size). "
            "This is the input format the tokenizer and the transformer pretrain on. The "
            "task commands (basecaller, demux, ...) chunk reads in memory with a different, "
            "full-coverage cut and do not read these files."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-i", "--input", required=True, metavar="DIR",
                   help="directory of .fast5 / .pod5 files")
    p.add_argument("-o", "--output", default=None, metavar="DIR",
                   help="output directory [default: a 'pretrain_chunk' directory next to --input]")
    p.add_argument("-m", "--modality", default="dna_r9",
                   choices=["dna_r9", "dna_r10", "rna_002", "rna_004"],
                   help="sets the trim threshold")
    p.add_argument("-c", "--chunk-size", type=int, default=16384, help="samples per chunk")
    p.add_argument("--overlap", type=int, default=2048, help="overlap between chunks, in samples")
    p.add_argument("-l", "--min-length", type=int, default=16384,
                   help="skip reads shorter than this many samples")
    p.add_argument("--max-chunks-per-file", type=int, default=100_000,
                   help="chunks per output shard")
    p.add_argument("-r", "--recursive", action="store_true",
                   help="also look in subdirectories of --input")
    p.add_argument("--resume", action="store_true",
                   help="skip input files that already have output shards")
    p.add_argument("-j", "--workers", type=int, default=1, metavar="N",
                   help="process N files in parallel (turns off per-file progress bars)")
    p.add_argument("--no-progress", action="store_true", help="no per-file progress bar")
    p.set_defaults(func=_run_train_prep)


def _run_train_prep(args):
    from brieno.data.chunk import build_pretrain_chunks

    build_pretrain_chunks(
        args.input,
        args.output,
        modality=args.modality,
        min_length=args.min_length,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        max_chunks_per_file=args.max_chunks_per_file,
        recursive=args.recursive,
        resume=args.resume,
        progress=not args.no_progress,
        workers=args.workers,
    )
    return 0


# -------------------------------------------------------------- brieno train merge
def _add_train_merge(subparsers):
    p = subparsers.add_parser(
        "merge",
        help="concatenate one dataset's chunk shards into a single .npy",
        description=(
            "Merge the shards 'brieno train prep' wrote for one dataset into the single "
            "array the tokenizer trainer reads. Done in two passes over memory-mapped "
            "files, so the corpus never has to fit in RAM. Name the output after the "
            "dataset: that name is the key used in the training config."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-i", "--input", required=True, metavar="DIR",
                   help="directory of .npy shards for ONE dataset")
    p.add_argument("-o", "--output", required=True, metavar="FILE",
                   help="output .npy file")
    p.add_argument("-r", "--recursive", action="store_true",
                   help="also look in subdirectories of --input")
    p.add_argument("-f", "--force", action="store_true", help="overwrite an existing output")
    p.add_argument("--no-progress", action="store_true", help="no progress bar")
    p.set_defaults(func=_run_train_merge)


def _run_train_merge(args):
    from brieno.data.merge import merge_shards

    merge_shards(args.input, args.output, recursive=args.recursive,
                 force=args.force, progress=not args.no_progress)
    return 0


# ----------------------------------------------------------- brieno train tokenizer
def _add_train_tokenizer(subparsers):
    p = subparsers.add_parser(
        "tokenizer",
        help="pretrain the signal tokenizer on chunked raw signal",
        description=(
            "Self-supervised pretraining of the residual-vector-quantization tokenizer: "
            "reconstruct the raw signal through the quantizer, minimising "
            "L1(reconstruction, signal) + vq_loss. No labels are used, so any corpus of "
            "raw reads works. Run under torchrun for multi-GPU training."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-c", "--config", required=True, metavar="JSON",
                   help="training config (see configs/ for an example)")
    p.add_argument("-s", "--seed", type=int, default=42, help="base seed; each rank adds its index")
    p.add_argument("-n", "--run-name", default=None, metavar="NAME",
                   help="output subdirectory under training.save_dir [default: a timestamp]")
    p.add_argument("--resume", default=None, metavar="CKPT",
                   help="continue from a checkpoint written by an earlier run")
    g = p.add_argument_group("config overrides (leave unset to use the config file)")
    g.add_argument("--lr", type=float, default=None)
    g.add_argument("--batch-size", type=int, default=None)
    g.add_argument("--epochs", type=int, default=None)
    g.add_argument("--commitment-cost", type=float, default=None)
    g.add_argument("--ema-decay", type=float, default=None)
    g.add_argument("--dropout", type=float, default=None,
                   help="sets encoder and decoder dropout together")
    p.set_defaults(func=_run_train_tokenizer)


def _run_train_tokenizer(args):
    from brieno.train.tokenizer import pretrain_tokenizer

    overrides = [
        ("training", "lr", args.lr),
        ("training", "batch_size", args.batch_size),
        ("training", "epochs", args.epochs),
        ("model", "commitment_cost", args.commitment_cost),
        ("model", "ema_decay", args.ema_decay),
        ("model", "encoder_dropout", args.dropout),
        ("model", "decoder_dropout", args.dropout),
    ]
    pretrain_tokenizer(args.config, seed=args.seed, run_name=args.run_name,
                       resume=args.resume, overrides=overrides)
    return 0


# ------------------------------------------------------------- brieno train encode
def _add_train_encode(subparsers):
    p = subparsers.add_parser(
        "encode",
        help="turn signal chunks into token ids with a trained tokenizer",
        description=(
            "Run a trained tokenizer over a merged signal file and write the discrete "
            "token ids the transformer pretrains on: (N, num_quantizers, seq_len) uint16. "
            "The tokenizer checkpoint carries its own architecture, so no config is needed."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-m", "--model", required=True, metavar="CKPT",
                   help="tokenizer checkpoint from 'brieno train tokenizer'")
    p.add_argument("-i", "--input", required=True, metavar="FILE",
                   help="merged signal .npy for one dataset")
    p.add_argument("-o", "--output", required=True, metavar="FILE", help="output token .npy")
    p.add_argument("-b", "--batch-size", type=int, default=512)
    p.add_argument("-j", "--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda", help="cuda, cuda:N or cpu")
    p.add_argument("-f", "--force", action="store_true", help="overwrite an existing output")
    p.add_argument("--no-progress", action="store_true", help="no progress bar")
    p.set_defaults(func=_run_train_encode)


def _run_train_encode(args):
    from brieno.data.encode import encode_dataset

    encode_dataset(args.model, args.input, args.output, batch_size=args.batch_size,
                   num_workers=args.num_workers, device=args.device,
                   force=args.force, progress=not args.no_progress)
    return 0


# -------------------------------------------------------- brieno train transformer
def _add_train_transformer(subparsers):
    p = subparsers.add_parser(
        "transformer",
        help="pretrain the token transformer with masked-token prediction",
        description=(
            "Self-supervised pretraining of the transformer on tokenized signal: spans of "
            "token positions are hidden and predicted from their context, in every "
            "quantizer layer at once. Run under torchrun for multi-GPU training. "
            "With model.input_mode 'latent' in the config the transformer reads the "
            "tokenizer's continuous latent instead of code ids and still predicts the code "
            "ids; that mode trains on the signal corpus and needs --tokenizer."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-c", "--config", required=True, metavar="JSON",
                   help="training config (see configs/ for an example)")
    p.add_argument("-s", "--seed", type=int, default=None,
                   help="base seed; each rank adds its index [default: training.seed]")
    p.add_argument("-n", "--run-name", default=None, metavar="NAME",
                   help="output subdirectory under training.save_dir [default: a timestamp]")
    p.add_argument("--resume", default=None, metavar="CKPT",
                   help="continue from a checkpoint written by an earlier run")
    p.add_argument("-t", "--tokenizer", default=None, metavar="CKPT",
                   help="latent input mode only: the frozen tokenizer that supplies the "
                        "input and the targets [default: the config's \"tokenizer\" entry]")
    g = p.add_argument_group("config overrides (leave unset to use the config file)")
    g.add_argument("--lr", type=float, default=None)
    g.add_argument("--batch-size", type=int, default=None)
    g.add_argument("--epochs", type=int, default=None)
    g.add_argument("--grad-accum-steps", type=int, default=None)
    p.set_defaults(func=_run_train_transformer)


def _run_train_transformer(args):
    from brieno.train.transformer import pretrain_transformer

    overrides = [
        ("training", "lr", args.lr),
        ("training", "batch_size", args.batch_size),
        ("training", "epochs", args.epochs),
        ("training", "grad_accum_steps", args.grad_accum_steps),
    ]
    pretrain_transformer(args.config, seed=args.seed, run_name=args.run_name,
                         resume=args.resume, overrides=overrides,
                         tokenizer=args.tokenizer)
    return 0


# ----------------------------------------------------------- brieno train classify
def _add_train_classify(subparsers):
    p = subparsers.add_parser(
        "classify",
        help="fine-tune a classification head (demultiplexing, species, ...)",
        description=(
            "Train one or more classification heads on top of the pretrained model. The "
            "number of classes comes from the dataset, so the same command serves barcode "
            "demultiplexing and species identification. Heads differ only in what is fed "
            "to them: A continuous latent, B1 quantized latent, B2 token ids, C1 tokens "
            "through the frozen transformer, C2 the same with the transformer fine-tuned, "
            "D1 the encoder architecture trained from scratch as the no-pretraining control."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-r", "--root", required=True, metavar="DIR", help="dataset root")
    p.add_argument("--setting", default="setting1", help="split directory under --root")
    p.add_argument("-f", "--fold", type=int, required=True)
    p.add_argument("-m", "--tokenizer", required=True, metavar="CKPT")
    p.add_argument("-t", "--transformer", default=None, metavar="CKPT",
                   help="required for heads C1 and C2")
    p.add_argument("--heads", default="C1",
                   help="comma-separated: A,B1,B2,C1,C2,D1 (C1 and C2 cannot share a job)")
    p.add_argument("-o", "--out-root", default=".", metavar="DIR")
    p.add_argument("--task-name", default="classify",
                   help="output directory prefix, e.g. demux or species")
    p.add_argument("-n", "--run-name", default=None, metavar="NAME")
    p.add_argument("--resume", default=None, metavar="CKPT", help="continue from a last.pt")
    g = p.add_argument_group("training")
    g.add_argument("--epochs", type=int, default=12)
    g.add_argument("--per-class", type=int, default=50000, help="chunks drawn per class per epoch")
    g.add_argument("--val-chunks", type=int, default=50000)
    g.add_argument("--batch", type=int, default=128)
    g.add_argument("--eval-batch", type=int, default=512)
    g.add_argument("--lr", type=float, default=5e-4, help="head learning rate")
    g.add_argument("--tf-lr", type=float, default=5e-5,
                   help="C2 only: a smaller rate for the pretrained transformer")
    g.add_argument("--wd", type=float, default=0.05)
    g.add_argument("--accum", type=int, default=1, help="gradient accumulation steps")
    g.add_argument("-j", "--workers", type=int, default=8)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--transformer-random", action="store_true",
                   help="keep the transformer architecture but not its weights: the "
                        "no-pretraining control for a fine-tuned head")
    p.set_defaults(func=_run_train_classify)


def _run_train_classify(args):
    from brieno.train.classify import finetune

    finetune(args.root, args.setting, args.fold, args.tokenizer,
             transformer_ckpt=args.transformer,
             heads=[h.strip() for h in args.heads.split(",") if h.strip()],
             out_root=args.out_root, task_name=args.task_name, run_name=args.run_name,
             epochs=args.epochs, per_class=args.per_class, val_chunks=args.val_chunks,
             batch=args.batch, eval_batch=args.eval_batch, lr=args.lr, tf_lr=args.tf_lr,
             weight_decay=args.wd, accum=args.accum, workers=args.workers,
             seed=args.seed, resume=args.resume,
             transformer_random=args.transformer_random)
    return 0


# ------------------------------------------------------------------- brieno train
def _add_train(subparsers):
    p = subparsers.add_parser(
        "train",
        help="developer tools: data preparation and model training",
        description="Building a Brieno model: data preparation and training.",
        epilog="Run 'brieno train <subcommand> --help' for the options of one subcommand.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="subcommand", metavar="<subcommand>")
    _add_train_prep(sub)
    _add_train_merge(sub)
    _add_train_tokenizer(sub)
    _add_train_encode(sub)
    _add_train_transformer(sub)
    _add_train_classify(sub)
    p.set_defaults(func=_print_help_and_fail(p))
    return p


def build_parser():
    parser = argparse.ArgumentParser(
        prog="brieno",
        description="Brieno: a foundation model for raw nanopore signal.",
        epilog="Run 'brieno <command> --help' for the options of one command.\n"
               "Documentation: https://github.com/guangzhaocs/Brieno",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-v", "--version", action=_VersionAction)
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")
    _add_train(subparsers)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 1
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nbrieno: interrupted", file=sys.stderr)
        return 130
    except (FileNotFoundError, FileExistsError, IsADirectoryError, NotADirectoryError,
            PermissionError, ValueError, KeyError) as exc:
        # Expected, actionable failures: a bad path, a malformed config, a mismatched
        # shard. Report them as a message, not as a traceback; anything else is a bug
        # and keeps its traceback.
        print(f"brieno: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
