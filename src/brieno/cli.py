"""The ``brieno`` command line.

Two layers, split by audience:

* **Using a trained model** — one top-level subcommand per task::

      brieno basecaller    raw signal -> basecalls
      brieno demux         raw signal -> one barcode per read
      brieno classifier    raw signal -> one species per read
      brieno modcaller     raw signal -> modified bases
      brieno download      fetch model weights

* **Building a model** — everything developer-facing lives under ``brieno train``::

      brieno train prep    raw FAST5/POD5 -> normalised signal chunks

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
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
