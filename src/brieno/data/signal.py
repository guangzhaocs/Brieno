"""Reading raw nanopore signal from FAST5 / POD5, and the per-read normalisation.

The readers expose one API over both formats::

    with get_reader(path) as reader:
        for read_id, signal_pa in reader.iter_reads():
            ...

Signal is always returned in **picoamperes** (``float32``): FAST5 is scaled with the
read's own calibration, POD5 is read through ``signal_pa``.

``ont-fast5-api`` and ``pod5`` are optional dependencies — each is imported only when a
file of that format is actually opened, so installing Brieno does not pull in both.
"""
from abc import ABC, abstractmethod

import numpy as np

__all__ = [
    "SignalReaderBase",
    "Fast5Reader",
    "Pod5Reader",
    "get_reader",
    "med_mad_normalisation",
    "trim",
]


class SignalReaderBase(ABC):
    """Unified API for reading signal out of one raw-signal file."""

    def __init__(self, filename):
        self.filename = str(filename)

    @abstractmethod
    def __enter__(self):
        """Open the file and return self."""

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close the file."""

    @abstractmethod
    def get_read_ids(self):
        """Return the read ids as a list (not a generator)."""

    @abstractmethod
    def get_signal(self, read_id):
        """Return one read's signal in pA as a float32 array."""

    @abstractmethod
    def iter_reads(self):
        """Yield ``(read_id, signal_pa)`` for every read in the file."""

    def get_total_reads(self):
        """Number of reads in the file (works for both formats)."""
        ids = self.get_read_ids()
        try:
            return len(ids)
        except TypeError:
            return sum(1 for _ in ids)


class Fast5Reader(SignalReaderBase):
    """Signal reader for FAST5 files (needs ``ont-fast5-api``)."""

    def __enter__(self):
        try:
            from ont_fast5_api.fast5_interface import get_fast5_file
        except ImportError as exc:                      # pragma: no cover - import guard
            raise ImportError(
                "Reading FAST5 needs the optional dependency `ont-fast5-api`: "
                "pip install 'brieno[fast5]'"
            ) from exc
        self.infile = get_fast5_file(self.filename, mode="r")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.infile.close()

    def get_read_ids(self):
        return self.infile.get_read_ids()

    def get_channel_info(self, read_id):
        """channel_number, digitisation, offset, range, sampling_rate."""
        return self.infile.get_read(read_id).get_channel_info()

    def get_signal(self, read_id):
        raw_signal = self.infile.get_read(read_id).get_raw_data(scale=True)
        return np.array(raw_signal, dtype=np.float32)

    def iter_reads(self):
        for read_id in self.get_read_ids():
            yield read_id, self.get_signal(read_id)


class Pod5Reader(SignalReaderBase):
    """Signal reader for POD5 files (needs ``pod5``)."""

    def __enter__(self):
        try:
            import pod5
        except ImportError as exc:                      # pragma: no cover - import guard
            raise ImportError(
                "Reading POD5 needs the optional dependency `pod5`: "
                "pip install 'brieno[pod5]'"
            ) from exc
        self.infile = pod5.Reader(self.filename)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.infile.close()

    def get_read_ids(self):
        return self.infile.read_ids

    def get_channel_info(self, read_id):
        """POD5 keeps calibration separate from the channel record."""
        r = next(self.infile.reads(selection=[read_id]))
        return {"channel_number": r.channel.channel, "well": r.channel.well}

    def get_signal(self, read_id):
        r = next(self.infile.reads(selection=[read_id]))
        return np.array(r.signal_pa, dtype=np.float32)

    def iter_reads(self):
        for read in self.infile.reads():
            yield read.read_id, np.array(read.signal_pa, dtype=np.float32)


SIGNAL_SUFFIXES = (".fast5", ".pod5")


def get_reader(filename):
    """Return the reader matching the file's extension.

    Raises ValueError for anything that is not ``.fast5`` or ``.pod5``.
    """
    name = str(filename)
    if name.endswith(".pod5"):
        return Pod5Reader(name)
    if name.endswith(".fast5"):
        return Fast5Reader(name)
    raise ValueError(f"Unsupported file format: {filename}")


def med_mad_normalisation(sig):
    """Robust per-read location and scale: ``(median, 1.4826 * MAD)``.

    ``sig`` is the pA-level current. Normalise with ``(sig - shift) / scale``.
    """
    shift = np.median(sig)
    mad = np.median(np.abs(sig - shift))
    scale = mad * 1.4826
    scale = max(scale, 1e-6)
    return shift, scale


def trim(signal, window_size=40, threshold=2.4, min_trim=10, min_elements=3,
         max_samples=8000, max_trim=0.3):
    """Index of the first sample after the open-pore / stall region at the read start.

    Verbatim port of bonito's ``reader.trim`` (window scan for a sustained excursion
    above ``threshold``, in normalised units). Returns ``min_trim`` when no peak is
    found or when trimming would eat more than ``max_trim`` of the read.
    """
    seen_peak = False
    num_windows = min(max_samples, len(signal)) // window_size

    for pos in range(num_windows):
        start = pos * window_size + min_trim
        end = start + window_size
        window = signal[start:end]
        if len(window[window > threshold]) > min_elements or seen_peak:
            seen_peak = True
            if window[-1] > threshold:
                continue
            if end >= min(max_samples, len(signal)) or end / len(signal) > max_trim:
                return min_trim
            return end

    return min_trim
