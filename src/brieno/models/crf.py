"""
CTC-CRF sequence distribution, the output layer Dorado and Bonito use
(bonito.crf.model.CTC_CRF, ONT Public License 1.0; re-implemented here so the Brieno
basecalling heads share exactly the same loss, scoring layout and beam search).

Layout (identical to bonito):  scores [T, N, C] with C = (n_base + 1) * n_base**state_len.
  Reshaped as [T, N, n_states, n_base + 1]: column 0 = stay in state s (blank), column 1 + a =
  move INTO state s having dropped oldest base a (the "(new_state, dropped_base)" convention).
  The emitted base of a move into s is its newest base (s % n_base).

Loss  = -logZ_target / |target| after GLOBAL normalisation (scores - logZ_all / T), both logZ
        computed by fused forward/backward lattice kernels on the GPU (fp32 only).
Decode: three decoders, in decreasing order of fidelity to Dorado:
        crf_beam_decode      koi.decode.beam_search (fp16 scores [N, T, C], beam 32,
                             beam_cut 100, blank_score 2.0) -> per-frame sequence / qstring /
                             moves, exactly what Dorado stitches. Needs the `ont-koi` wheel.
        crf_fcd_beam_decode  fast_ctc_decode.crf_beam_search (open-source Rust, CPU) on the
                             posterior transition probabilities. Runs anywhere.
        crf_viterbi_decode   best path, no beam.

Dorado R9.4.1 sup settings: state_len 5, LinearCRF tanh x scale 5.0, blank_score 2.0 fixed.

Backend (`BACKEND` says which): `koi.ctc` when the `ont-koi` wheel is installed (x86_64
only), otherwise the cupy port of the same open-source seqdist kernels in
`brieno/models/seqdist_cupy.py`, which compiles at run time on any CUDA platform.

The fast_ctc_decode beam at width 32 agrees with the koi beam on 99.7 % of bases and scores
about 0.09 points lower in chunk-level identity. Numbers decoded with one are comparable
with each other; numbers decoded with different decoders must not share a column.
"""
import numpy as np
import torch
import torch.nn as nn

try:                                   # ONT's closed CUDA kernels (x86_64 wheel only)
    from koi.ctc import SequenceDist, Max, Log, semiring
    from koi.ctc import logZ_cu, viterbi_alignments, logZ_cu_sparse, bwd_scores_cu_sparse, fwd_scores_cu_sparse
    BACKEND = "koi"
except ImportError:
    try:                               # cupy port of the same (seqdist) kernels: any CUDA platform
        from brieno.models.seqdist_cupy import SequenceDist, Max, Log, semiring
        from brieno.models.seqdist_cupy import logZ_cu, viterbi_alignments, logZ_cu_sparse, bwd_scores_cu_sparse, fwd_scores_cu_sparse
        BACKEND = "cupy"
    except ImportError:                # no CUDA backend: importing this module must still work
        BACKEND = None
        SequenceDist = object

        def _no_backend(*_a, **_k):
            raise ImportError(
                "The CRF loss and Viterbi decoding need a CUDA backend: install `ont-koi` "
                "(x86_64) or `cupy` (any CUDA platform, including aarch64). "
                "Everything else in brieno.models works without one."
            )

        Max = Log = semiring = None
        logZ_cu = viterbi_alignments = logZ_cu_sparse = _no_backend
        bwd_scores_cu_sparse = fwd_scores_cu_sparse = _no_backend

ALPHABET = ["N", "A", "C", "G", "T"]


class CTC_CRF(SequenceDist):
    """Verbatim semantics of bonito.crf.model.CTC_CRF (koi backed)."""

    def __init__(self, state_len=5, alphabet=ALPHABET):
        super().__init__()
        self.alphabet = list(alphabet)
        self.state_len = int(state_len)
        self.n_base = len(self.alphabet[1:])
        self.idx = torch.cat([
            torch.arange(self.n_base ** self.state_len)[:, None],
            torch.arange(self.n_base ** self.state_len).repeat_interleave(self.n_base).reshape(self.n_base, -1).T
        ], dim=1).to(torch.int32)

    @property
    def n_states(self):
        return self.n_base ** self.state_len

    def n_score(self):
        return len(self.alphabet) * self.n_states

    def logZ(self, scores, S: semiring = Log):
        T, N, _ = scores.shape
        Ms = scores.reshape(T, N, -1, len(self.alphabet))
        alpha_0 = Ms.new_full((N, self.n_states), S.one)
        beta_T = Ms.new_full((N, self.n_states), S.one)
        return logZ_cu_sparse(Ms, self.idx, alpha_0, beta_T, S)

    def normalise(self, scores):
        return scores - self.logZ(scores)[:, None] / len(scores)

    def forward_scores(self, scores, S: semiring = Log):
        T, N, _ = scores.shape
        Ms = scores.reshape(T, N, -1, self.n_base + 1)
        alpha_0 = Ms.new_full((N, self.n_states), S.one)
        return fwd_scores_cu_sparse(Ms, self.idx, alpha_0, S, K=1)

    def backward_scores(self, scores, S: semiring = Log):
        T, N, _ = scores.shape
        Ms = scores.reshape(T, N, -1, self.n_base + 1)
        beta_T = Ms.new_full((N, self.n_states), S.one)
        return bwd_scores_cu_sparse(Ms, self.idx, beta_T, S, K=1)

    def viterbi(self, scores):
        """scores [T,N,C] fp32 -> paths [T,N] (0 = stay, 1..4 = emitted base id)."""
        traceback = self.posteriors(scores, Max)
        a_traceback = traceback.argmax(2)
        moves = (a_traceback % len(self.alphabet)) != 0
        paths = 1 + (torch.div(a_traceback, len(self.alphabet), rounding_mode="floor") % self.n_base)
        return torch.where(moves, paths, 0)

    def path_to_str(self, path):
        alphabet = np.frombuffer(''.join(self.alphabet).encode(), dtype='u1')
        seq = alphabet[path[path != 0]]
        return seq.tobytes().decode()

    def prepare_ctc_scores(self, scores, targets):
        """targets [N, U] long, CTC convention (1..4 = ACGT, 0 = pad)."""
        targets = torch.clamp(targets - 1, 0)
        T, N, C = scores.shape
        scores = scores.to(torch.float32)
        n = targets.size(1) - (self.state_len - 1)
        stay_indices = sum(
            targets[:, i:n + i] * self.n_base ** (self.state_len - i - 1)
            for i in range(self.state_len)
        ) * len(self.alphabet)
        move_indices = stay_indices[:, 1:] + targets[:, :n - 1] + 1
        stay_scores = scores.gather(2, stay_indices.expand(T, -1, -1))
        move_scores = scores.gather(2, move_indices.expand(T, -1, -1))
        return stay_scores, move_scores

    def ctc_loss(self, scores, targets, target_lengths, loss_clip=None, reduction='mean', normalise_scores=True):
        """scores [T,N,C] (any float dtype; cast to fp32), targets [N,U], target_lengths [N]."""
        scores = scores.to(torch.float32)
        if normalise_scores:
            scores = self.normalise(scores)
        stay_scores, move_scores = self.prepare_ctc_scores(scores, targets)
        logz = logZ_cu(stay_scores, move_scores, target_lengths + 1 - self.state_len)
        loss = - (logz / target_lengths)
        if loss_clip:
            loss = torch.clamp(loss, 0.0, loss_clip)
        if reduction == 'mean':
            return loss.mean()
        elif reduction in ('none', None):
            return loss
        raise ValueError(f'Unknown reduction type {reduction}')

    def ctc_viterbi_alignments(self, scores, targets, target_lengths):
        stay_scores, move_scores = self.prepare_ctc_scores(scores, targets)
        return viterbi_alignments(stay_scores, move_scores, target_lengths + 1 - self.state_len)


class LinearCRF(nn.Module):
    """bonito LinearCRFEncoder with expand_blanks: Linear(in -> n_base**(state_len+1)), tanh, x scale,
    then a constant blank_score column is prepended per state -> [T, N, (n_base+1) * n_base**state_len]."""

    def __init__(self, insize, state_len=5, n_base=4, scale=5.0, blank_score=2.0, bias=True):
        super().__init__()
        self.state_len, self.n_base, self.scale, self.blank_score = state_len, n_base, scale, blank_score
        self.linear = nn.Linear(insize, n_base ** (state_len + 1), bias=bias)

    def forward(self, x):                      # x [B, T, in] -> scores [T, B, C]
        s = torch.tanh(self.linear(x)) * self.scale
        B, T, C = s.shape
        s = torch.nn.functional.pad(s.view(B, T, C // self.n_base, self.n_base), (1, 0), value=self.blank_score)
        return s.view(B, T, -1).transpose(0, 1).contiguous()

    @property
    def n_out(self):
        return (self.n_base + 1) * self.n_base ** self.state_len


def strip_blank_column(scores_TNC, n_base=4):
    """[T, N, (n_base+1) * n_states] (blank column per state, as the loss wants) -> [T, N, n_base * n_states] move scores only.
    koi.decode.beam_search expects the blank-free layout (C = n_base ** (state_len + 1), cf. its HP_STATES table) and adds the
    constant blank_score itself — exactly what bonito's koi-optimised LinearCRF (expand_blanks=False) feeds it."""
    T, N, C = scores_TNC.shape
    return scores_TNC.view(T, N, C // (n_base + 1), n_base + 1)[..., 1:].reshape(T, N, -1)


@torch.no_grad()
def crf_beam_decode(scores_TNC, beam_width=32, beam_cut=100.0, scale=1.0, offset=0.0, blank_score=2.0):
    """Dorado-style decode of raw (un-normalised) CRF scores [T, N, 5 * 4**L]. Returns (seqs, qstrings, moves[N,T] int8).
    koi needs fp16, contiguous, layout [N, T, 4**(L+1)] WITHOUT the blank column, on a CUDA device with an explicit index."""
    try:
        from koi.decode import beam_search as koi_beam_search, to_str as koi_to_str
    except ImportError as e:   # closed-source, x86_64 wheel only
        raise ImportError("koi.decode.beam_search is unavailable on this platform (BACKEND=%s); "
                          "use crf_fcd_beam_decode or crf_viterbi_decode" % BACKEND) from e
    scores = strip_blank_column(scores_TNC).transpose(0, 1).to(torch.float16).contiguous()
    seq, qs, mv = koi_beam_search(scores, beam_width=beam_width, beam_cut=beam_cut, scale=scale,
                                  offset=offset, blank_score=blank_score)
    seqs = [koi_to_str(seq[i]) for i in range(seq.shape[0])]
    qstrs = [koi_to_str(qs[i]) for i in range(qs.shape[0])]
    return seqs, qstrs, mv.numpy()


@torch.no_grad()
def crf_viterbi_decode(seqdist: CTC_CRF, scores_TNC):
    """bonito SeqdistModel.decode_batch: posteriors -> log -> viterbi -> strings."""
    post = seqdist.posteriors(scores_TNC.to(torch.float32)) + 1e-8
    paths = seqdist.viterbi(post.log()).to(torch.int16).T.cpu().numpy()
    return [seqdist.path_to_str(p) for p in paths]

def crf_transition_probs(seqdist: CTC_CRF, scores_TNC, betas=None):
    """bonito (pre-koi) CTC_CRF.compute_transition_probs: posterior transition probabilities for fast_ctc_decode.
    scores [T,N,C] in the (new_state, dropped_base) layout, betas [T+1,N,S] (backward_scores) ->
    trans [T,N,S,1+n_base] probs in the (old_state, emitted_base) layout, init [N,S] = softmax(betas[0])."""
    scores = scores_TNC.to(torch.float32)
    if betas is None: betas = seqdist.backward_scores(scores)
    T, N, C = scores.shape; nb = seqdist.n_base
    lt = scores.reshape(T, N, -1, nb + 1) + betas[1:, :, :, None]
    lt = torch.cat([lt[:, :, :, [0]], lt[:, :, :, 1:].transpose(3, 2).reshape(T, N, -1, nb)], dim=-1)
    return torch.softmax(lt, dim=-1), torch.softmax(betas[0], dim=-1)


@torch.no_grad()
def crf_fcd_beam_decode(seqdist: CTC_CRF, scores_TNC, beam_size=32, beam_cut=0.0, threads=8, with_path=False):
    """CRF beam search with fast_ctc_decode.crf_beam_search (open-source Rust, CPU; x86_64 + aarch64 wheels) on the
    bonito posterior transition probabilities. The decoder to use where koi.decode is unavailable.

    with_path=False -> list of decoded strings (chunk-level scoring).
    with_path=True  -> list of (seq, path); `path` is one FRAME INDEX per emitted base, strictly increasing in [0,T):
                       rebuilding a [T] per-frame array with arr[path] = bases and collapsing on `!= 0` reproduces
                       seq byte for byte. Read-level stitching happens in frame space, so it needs this."""
    from concurrent.futures import ThreadPoolExecutor
    from fast_ctc_decode import crf_beam_search
    trans, init = crf_transition_probs(seqdist, scores_TNC)
    trans = trans.permute(1, 0, 2, 3).contiguous().cpu().numpy(); init = init.cpu().numpy()   # [N,T,S,5], [N,S]
    alphabet = "".join(seqdist.alphabet)
    def fn(i):
        seq, path = crf_beam_search(trans[i], init[i], alphabet, beam_size=beam_size, beam_cut_threshold=beam_cut)
        return (seq, path) if with_path else seq
    with ThreadPoolExecutor(threads) as pool:
        return list(pool.map(fn, range(trans.shape[0])))


def crf_fcd_frames(seqdist: CTC_CRF, scores_TNC, beam_size=32, beam_cut=0.0, threads=8):
    """fcd CRF beam search returned in koi.decode.beam_search's PER-FRAME layout: (seq, qstring) int8 [N, T].

    koi gives (seq, qstring, moves) per frame, and read-level inference stitches those across overlapping chunks
    before collapsing with `seq != 0`.  fcd gives (string, frame-index-per-base) instead, so scatter the bases back
    onto the frame axis.  fcd has no calibrated q-scores -> the qstring is a constant fill ("+"): accuracy is scored
    on the SEQUENCE only, so no reported number is affected, but the quality column of a fastq written from this
    decoder is meaningless and must not be used."""
    import numpy as np
    T = int(scores_TNC.shape[0])
    out = crf_fcd_beam_decode(seqdist, scores_TNC, beam_size=beam_size, beam_cut=beam_cut, threads=threads, with_path=True)
    seq = np.zeros((len(out), T), dtype=np.int8); qs = np.zeros((len(out), T), dtype=np.int8)
    for i, (s, path) in enumerate(out):
        if not len(s): continue
        p = np.asarray(path, dtype=np.int64)
        seq[i, p] = np.frombuffer(s.encode(), dtype=np.int8)
        qs[i, p] = ord("+")
    return seq, qs
