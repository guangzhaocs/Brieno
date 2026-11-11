"""
Drop-in replacement for the `koi.ctc` CUDA functions on platforms without an `ont-koi` wheel (e.g.
aarch64 GPU nodes), built on cupy run-time-compiled kernels.

Provenance: `koi.ctc` (ONT, closed CUDA binary) is a renamed copy of David Page's open-source `seqdist`
(MIT, https://github.com/davidcpage/seqdist, v0.0.3) — same `LogZ` autograd class, same kernel names
(`fwd_bwd_logspace`, `logZ_fwd`, `fwd_scores`, `bwd_scores`), same `Log` / `Max` semirings.  The CUDA
sources below are seqdist's `cuda/ctc_simple.cu` and `cuda/sparse_logZ.cu` verbatim; the python glue is
seqdist's with koi's argument conventions (K=1, fp32), so `brieno/models/crf.py` can import
either backend unchanged.  Kernels are compiled by cupy/nvrtc at first use for whatever GPU is present
(no pre-built binary -> no aarch64 problem) and launched on torch's current CUDA stream.

Exports (koi.ctc names): semiring, Log, Max, grad, SequenceDist, logZ_cu, viterbi_alignments,
                         fwd_scores_cu_sparse, bwd_scores_cu_sparse, logZ_fwd_cu, logZ_cu_sparse
Not provided: koi.decode.beam_search (closed C++/CUDA beam search) -> decode with
              crf_fcd_beam_decode or crf_viterbi_decode (brieno/models/crf.py) on this backend.
"""
from collections import namedtuple
from functools import lru_cache

import cupy as cp
import torch

# ----------------------------------------------------------------------------- semirings (== koi.ctc)
semiring = namedtuple('semiring', ('zero', 'one', 'mul', 'sum', 'dsum'))


def max_grad(x, dim=0):
    return torch.zeros_like(x).scatter_(dim, x.argmax(dim, True), 1.0)


Log = semiring(zero=-1e38, one=0., mul=torch.add, sum=torch.logsumexp, dsum=torch.softmax)
Max = semiring(zero=-1e38, one=0., mul=torch.add, sum=(lambda x, dim=0: torch.max(x, dim=dim)[0]), dsum=max_grad)


def grad(f, x):
    x = x.detach().requires_grad_()
    with torch.enable_grad():
        y = f(x)
        return torch.autograd.grad(y, x)[0].detach()


class SequenceDist():
    def __init__(self):
        pass

    def logZ(self, scores, S: semiring = Log):
        raise NotImplementedError

    def viterbi(self, scores):
        raise NotImplementedError

    def ctc_loss(self, scores, targets, target_lengths):
        raise NotImplementedError

    def posteriors(self, scores, S: semiring = Log):
        f = lambda x: self.logZ(x, S).sum()
        return grad(f, scores)


# ----------------------------------------------------------------------------- CUDA sources (seqdist, MIT)
_CTC_SIMPLE_CU = r"""
__device__ __forceinline__ FLOAT max2(FLOAT a, FLOAT a1) {
    return a > a1 ? a : a1;
}

__device__ __forceinline__ FLOAT logsumexp2(FLOAT a, FLOAT a1) {
    return a > a1 ? a + log1p(exp(a1 - a)) : a1 + log1p(exp(a - a1));
}

__device__ __forceinline__ FLOAT add(FLOAT a, FLOAT b) {return a + b;}
__device__ __forceinline__ FLOAT mul(FLOAT a, FLOAT b) {return a * b;}

extern "C" __global__ void fwd_bwd_logspace(
    FLOAT* __restrict__ alpha, FLOAT* __restrict__ beta_T,
    FLOAT* __restrict__ beta_stay, FLOAT* __restrict__ beta_move,
    const FLOAT* __restrict__ stay_scores, const FLOAT* __restrict__ move_scores,
    int T, int N, int L
) {
    int bx = blockIdx.x, tx = threadIdx.x;
    if (tx >= L) return;
    extern __shared__ FLOAT smem[];
    if (blockIdx.y == 0) {
        FLOAT a = ZERO, a1 = ZERO;
        a = alpha[bx * L + tx];
        for (int t = 0; t < T; t++) {
            FLOAT *buf = smem + (t % 2) * blockDim.x;
            buf[tx] = a; __syncthreads();
            if (tx > 0) {a1 = MUL(move_scores[(t * N + bx) * (L - 1) + tx - 1], buf[tx - 1]);}
            a = SUM(MUL(stay_scores[(t * N + bx) * L + tx], a), a1);
            alpha[((t + 1) * N + bx) * L + tx] = a;
        }
    }
    else {
        FLOAT b = ZERO, b1 = ZERO;
        b = beta_T[bx * L + tx];
        for (int t = T; t > 0; t--) {
            FLOAT *buf = smem + (t % 2) * blockDim.x;
            buf[tx] = b; __syncthreads();
            if (tx < L - 1) {
                b1 = MUL(buf[tx + 1], move_scores[(((t - 1) * N + bx) * (L - 1)) + tx]);
                beta_move[((t - 1) * N + bx) * L + tx] = b1;
            }
            b = MUL(b, stay_scores[(((t - 1) * N + bx) * L) + tx]);
            beta_stay[((t - 1) * N + bx) * L + tx] = b;
            b = SUM(b, b1);
        }
    }
  }

extern "C" __global__ void fwd_bwd_logspace_loop(
    FLOAT* __restrict__ alpha, FLOAT* __restrict__ beta,
    FLOAT* __restrict__ beta_stay, FLOAT* __restrict__ beta_move,
    const FLOAT* __restrict__ stay_scores, const FLOAT* __restrict__ move_scores,
    int T, int N, int L
) {
    int bx = blockIdx.x, tx = threadIdx.x;
    if (blockIdx.y == 0) {
        FLOAT a;
        for (int t = 0; t < T; t++) {
            for (int j = tx; j < L; j += blockDim.x) {
                a = (j > 0) ? MUL(move_scores[(t * N + bx) * (L - 1) + j - 1], alpha[(t * N + bx) * L + j - 1]) : ZERO;
                alpha[((t + 1) * N + bx) * L + j] = SUM(MUL(stay_scores[(t * N + bx) * L + j], alpha[(t * N + bx) * L + j]), a);
            }
            __syncthreads();
        }
    }
    else {
        FLOAT b, b1;
        for (int t = T; t > 0; t--) {
            for (int j = L - blockDim.x + tx; j >= 0; j -= blockDim.x) {
                b1 = ZERO;
                if (j < L - 1) {
                    b1 = MUL(beta[(t * N + bx) * L + j + 1], move_scores[(((t - 1) * N + bx) * (L - 1)) + j]);
                    beta_move[((t - 1) * N + bx) * L + j] = b1;
                }
                b = MUL(beta[(t * N + bx) * L + j], stay_scores[(((t - 1) * N + bx) * L) + j]);
                beta_stay[((t - 1) * N + bx) * L + j] = b;
                beta[((t - 1) * N + bx) * L + j] = SUM(b, b1);
            }
            __syncthreads();
        }
    }
  }
"""

_SPARSE_LOGZ_CU = r"""
__device__ __forceinline__ FLOAT add(FLOAT a, FLOAT b) {return a + b;}
__device__ __forceinline__ FLOAT max_(FLOAT *s) {
    FLOAT mx = s[0];
    for (int j = 1; j < NZ; j++) {
        mx = mx > s[j] ? mx : s[j];
    }
    return mx;
}
__device__ __forceinline__ FLOAT logsumexp(FLOAT *s) {
    FLOAT mx = max_(s);
    FLOAT res = exp(s[0] - mx);
    for (int j = 1; j < NZ; j++) {
        res += exp(s[j] - mx);
    }
    return log(res) + mx;
}

extern "C" __global__ void logZ_fwd(
    FLOAT* __restrict__ logZ,
    FLOAT* __restrict__ Ms_grad,
    const FLOAT* __restrict__ Ms,
    const FLOAT* __restrict__ v0,
    const FLOAT* __restrict__ vT,
    const int* __restrict__ idx,
    int T, int N, int C
) {
    int bx = blockIdx.x;
    int tx = threadIdx.x * K;
    if (tx >= C) return;
    extern __shared__ FLOAT smem[];

    FLOAT a[K];
    for (int k = 0; k < K; k++) {
        a[k] = v0[bx * C + tx + k];
    }
    __syncthreads();

    FLOAT s[NZ];
    for (int t = 0; t < T; t++) {
        FLOAT *buf = smem + (t % 2) * blockDim.x * K;
        for (int k = 0; k < K; k++) {
            buf[tx+k] = a[k];
        }
        __syncthreads();
        int i = (t * N + bx) * C * NZ;
        for (int k = 0; k < K; k++) {
            for (int j = 0; j < NZ; j++) {
                s[j] = MUL(buf[idx[(tx + k) * NZ + j]], Ms[i + (tx + k) * NZ + j]);
                Ms_grad[i + (tx + k) * NZ + j] = s[j];
            }
            a[k] = SUM(s);
        }
    }

    for (int k = 0; k < K; k++) {
        logZ[bx * C + tx + k] = MUL(a[k], vT[bx * C + tx + k]);
    }
}

extern "C" __global__ void fwd_scores(
    FLOAT* __restrict__ alphas,
    const FLOAT* __restrict__ Ms,
    const FLOAT* __restrict__ v0,
    const int* __restrict__ idx,
    int T, int N, int C
) {
    int bx = blockIdx.x;
    int tx = threadIdx.x * K;
    if (tx >= C) return;
    extern __shared__ FLOAT smem[];

    FLOAT a[K];
    for (int k = 0; k < K; k++) {
        a[k] = v0[bx * C + tx + k];
        alphas[(bx * C) + tx + k] = a[k];
    }
    __syncthreads();

    FLOAT s[NZ];
    for (int t = 0; t < T; t++) {
        FLOAT *buf = smem + (t % 2) * blockDim.x * K;
        for (int k = 0; k < K; k++) {
            buf[tx+k] = a[k];
        }
        __syncthreads();
        int i = (t * N + bx) * C;
        for (int k = 0; k < K; k++) {
            for (int j = 0; j < NZ; j++) {
                s[j] = MUL(buf[idx[(tx + k) * NZ + j]], Ms[(i + tx + k) * NZ + j]);
            }
            a[k] = SUM(s);
            alphas[(i + N * C) + tx + k] = a[k];
        }
    }
}

extern "C" __global__ void bwd_scores(
    FLOAT* __restrict__ betas,
    const FLOAT* __restrict__ Ms,
    const FLOAT* __restrict__ vT,
    const int* __restrict__ idx_T,
    int T, int N, int C
) {
    int bx = blockIdx.x;
    int tx = threadIdx.x * K;
    if (tx >= C) return;
    extern __shared__ FLOAT smem[];

    FLOAT a[K];
    for (int k = 0; k < K; k++) {
        a[k] = vT[bx * C + tx + k];
        betas[(T * N + bx) * C + tx + k] = a[k];
    }
    __syncthreads();

    FLOAT s[NZ];
    for (int t = T - 1; t >= 0; t--) {
        FLOAT *buf = smem + (t % 2) * blockDim.x * K;
        for (int k = 0; k < K; k++) {
            buf[tx+k] = a[k];
        }
        __syncthreads();
        int i = (t * N + bx) * C;
        for (int k = 0; k < K; k++) {
            for (int j = 0; j < NZ; j++) {
                int ix = idx_T[(tx + k) * NZ + j];
                s[j] = MUL(buf[ix / NZ], Ms[i * NZ + ix]);
            }
            a[k] = SUM(s);
            betas[i + tx + k] = a[k];
        }
    }
}
"""

_FLOAT = {torch.float32: 'float', torch.float64: 'double'}
_BYTES = {torch.float32: 4, torch.float64: 8}


def _kernel(src, name, **macros):
    code = '\n'.join('#define {!s} {!s}'.format(k, v) for k, v in macros.items()) + '\n' + src
    return cp.RawKernel(code, name)


@lru_cache(None)
def _simple_kernel(name, dtype, S):
    # S is a namedtuple with function fields: hashable by identity, fine for the cache (only Log / Max are used)
    return _kernel(_CTC_SIMPLE_CU, name, FLOAT=_FLOAT[dtype], SUM='logsumexp2' if S is Log else 'max2',
                   MUL='add', ZERO='{:E}'.format(S.zero))


@lru_cache(None)
def _sparse_kernel(name, dtype, S, NZ, K):
    return _kernel(_SPARSE_LOGZ_CU, name, FLOAT=_FLOAT[dtype], MUL='add', ZERO='{:E}'.format(S.zero),
                   SUM='logsumexp' if S is Log else 'max_', NZ=NZ, K=K)


def _launch(kern, grid, block, args, device, shared_mem=0):
    """Launch on torch's current stream of `device` so ordering with the surrounding torch ops holds."""
    if block[0] > 1024:
        raise RuntimeError(f'block {block} exceeds 1024 threads')
    index = device.index if device.index is not None else torch.cuda.current_device()
    with cp.cuda.Device(index):
        with cp.cuda.ExternalStream(torch.cuda.current_stream(index).cuda_stream):
            kern(grid, block, args, shared_mem=shared_mem)


def _check(x, S):
    if x.dtype not in _FLOAT:
        raise NotImplementedError('Only fp32 / fp64 supported')
    if S is not Log and S is not Max:
        raise NotImplementedError('Semiring not supported')


# ----------------------------------------------------------------------------- target lattice (koi logZ_cu)
class LogZ(torch.autograd.Function):
    @staticmethod
    def forward(ctx, stay_scores, move_scores, target_lengths, fwd_bwd_impl, S: semiring):
        T, N, L = stay_scores.shape

        alpha = stay_scores.new_full((T + 1, N, L), S.zero)
        alpha[0, :, 0] = S.one

        beta_stay = stay_scores.new_full((T, N, L), S.zero)
        beta_move = stay_scores.new_full((T, N, L), S.zero)
        beta_T = stay_scores.new_full((N, L), S.zero)
        beta_T[torch.arange(N), target_lengths - 1] = S.one

        fwd_bwd_impl(alpha, beta_T, beta_stay, beta_move, stay_scores, move_scores, S)

        g = S.dsum(torch.cat([S.mul(alpha[:-1], beta_stay), S.mul(alpha[:-1], beta_move)], dim=2), dim=2)

        ctx.save_for_backward(g.reshape(T, N, 2, L))
        return dot(alpha[-1], beta_T, S)

    @staticmethod
    def backward(ctx, grad):
        g = ctx.saved_tensors[0] * grad[None, :, None, None]
        return g[:, :, 0], g[:, :, 1, :-1], None, None, None


def dot(x, y, S=Log, dim=-1):
    return S.sum(S.mul(x, y), dim=dim)


def _simple_lattice_fwd_bwd_cu_loop(alpha, beta_T, beta_stay, beta_move, stay_scores, move_scores, S: semiring,
                                    max_block_size=1024):
    T, N, L = stay_scores.shape
    block_size = min(L, max_block_size)
    beta = alpha.new_full(alpha.shape, S.zero)
    beta[-1] = beta_T
    stay_scores, move_scores = stay_scores.contiguous(), move_scores.contiguous()
    _launch(_simple_kernel('fwd_bwd_logspace_loop', stay_scores.dtype, S), (N, 2, 1), (block_size, 1, 1),
            (alpha.data_ptr(), beta.data_ptr(), beta_stay.data_ptr(), beta_move.data_ptr(),
             stay_scores.data_ptr(), move_scores.data_ptr(), T, N, L), stay_scores.device)


def _simple_lattice_fwd_bwd_cu(alpha, beta_T, beta_stay, beta_move, stay_scores, move_scores, S: semiring):
    _check(stay_scores, S)
    T, N, L = stay_scores.shape
    if L > 1024:  # exceeds max threads per block
        return _simple_lattice_fwd_bwd_cu_loop(alpha, beta_T, beta_stay, beta_move, stay_scores, move_scores, S)
    stay_scores, move_scores = stay_scores.contiguous(), move_scores.contiguous()
    _launch(_simple_kernel('fwd_bwd_logspace', stay_scores.dtype, S), (N, 2, 1), (L, 1, 1),
            (alpha.data_ptr(), beta_T.data_ptr(), beta_stay.data_ptr(), beta_move.data_ptr(),
             stay_scores.data_ptr(), move_scores.data_ptr(), T, N, L), stay_scores.device,
            shared_mem=2 * _BYTES[stay_scores.dtype] * L)


def logZ_cu(stay_scores, move_scores, target_lengths, S: semiring = Log):
    return LogZ.apply(stay_scores, move_scores, target_lengths, _simple_lattice_fwd_bwd_cu, S)


def viterbi_alignments(stay_scores, move_scores, target_lengths):
    target_lengths = target_lengths.to(stay_scores.device)
    stay_scores, move_scores = stay_scores.detach().requires_grad_(), move_scores.detach().requires_grad_()
    logZ_cu(stay_scores, move_scores, target_lengths, Max).sum().backward()
    alignments = stay_scores.grad.clone()
    alignments[:, :, :-1] += move_scores.grad
    return alignments


# ----------------------------------------------------------------------------- full state lattice (koi *_sparse)
def fwd_scores_cu_sparse(Ms, idx, v0, S: semiring = Log, K=1):
    _check(Ms, S)
    T, N, C, NZ = Ms.shape
    Ms = Ms.contiguous()
    alphas = Ms.new_full((T + 1, N, C), S.zero)
    idx = idx.to(dtype=torch.int, device=Ms.device).contiguous()
    _launch(_sparse_kernel('fwd_scores', Ms.dtype, S, NZ, K), (N, 1, 1), (C // K, 1, 1),
            (alphas.data_ptr(), Ms.data_ptr(), v0.contiguous().data_ptr(), idx.data_ptr(), T, N, C), Ms.device,
            shared_mem=2 * _BYTES[Ms.dtype] * C)
    return alphas


def bwd_scores_cu_sparse(Ms, idx, vT, S: semiring = Log, K=1):
    _check(Ms, S)
    T, N, C, NZ = Ms.shape
    Ms = Ms.contiguous()
    betas = Ms.new_full((T + 1, N, C), S.zero)
    idx_T = idx.flatten().argsort().to(dtype=torch.int, device=Ms.device).contiguous()  # transpose
    _launch(_sparse_kernel('bwd_scores', Ms.dtype, S, NZ, K), (N, 1, 1), (C // K, 1, 1),
            (betas.data_ptr(), Ms.data_ptr(), vT.contiguous().data_ptr(), idx_T.data_ptr(), T, N, C), Ms.device,
            shared_mem=2 * _BYTES[Ms.dtype] * C)
    return betas


def logZ_fwd_cu(Ms, idx, v0, vT, S: semiring = Log, K=1):
    _check(Ms, S)
    T, N, C, NZ = Ms.shape
    assert idx.shape == (C, NZ)
    Ms = Ms.contiguous()
    idx = idx.to(dtype=torch.int, device=Ms.device).contiguous()
    Ms_grad = Ms.new_full((T, N, C, NZ), S.zero)
    logZ = Ms.new_full((N, C), S.zero)
    _launch(_sparse_kernel('logZ_fwd', Ms.dtype, S, NZ, K), (N, 1, 1), (C // K, 1, 1),
            (logZ.data_ptr(), Ms_grad.data_ptr(), Ms.data_ptr(), v0.contiguous().data_ptr(),
             vT.contiguous().data_ptr(), idx.data_ptr(), T, N, C), Ms.device,
            shared_mem=2 * _BYTES[Ms.dtype] * C)
    return S.sum(logZ, dim=1), Ms_grad


class _LogZ(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Ms, idx, v0, vT, S: semiring, K):
        idx = idx.to(device=Ms.device)
        logZ, Ms_grad = logZ_fwd_cu(Ms, idx, v0, vT, S, K)
        ctx.save_for_backward(Ms_grad, Ms, idx, vT)
        ctx.semiring = S
        ctx.K = K
        return logZ

    @staticmethod
    def backward(ctx, grad):
        Ms_grad, Ms, idx, vT = ctx.saved_tensors
        S, K = ctx.semiring, ctx.K
        T, N, C, NZ = Ms.shape
        betas = bwd_scores_cu_sparse(Ms, idx, vT, S, K=K)
        Ms_grad = S.mul(Ms_grad, betas[1:, :, :, None])
        Ms_grad = S.dsum(Ms_grad.reshape(T, N, -1), dim=2).reshape(T, N, C, NZ)
        return grad[None, :, None, None] * Ms_grad, None, None, None, None, None


def logZ_cu_sparse(Ms, idx, v0, vT, S: semiring = Log, K=1):
    return _LogZ.apply(Ms, idx, v0, vT, S, K)
