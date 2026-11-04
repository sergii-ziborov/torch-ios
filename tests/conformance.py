"""Runs a few hundred PyTorch operations on fixed inputs and writes the results as JSON.

Run it on iOS and on a reference machine with the same torch version, then compare the
two files with tests/compare.py. Inputs come from formulas, not from a random generator,
so both sides see the same numbers. Where a result is only defined up to sign or order
(eigenvectors, SVD, QR), the case records what is unique: values, magnitudes and
reconstructions.

    python conformance.py out.json
"""

import io
import json
import math
import sys
import traceback

import torch
import torch.nn.functional as F
from torch import nn

CASES = {}


def case(name):
    def register(function):
        CASES[name] = function
        return function
    return register


def det(*shape, k=0.0, dtype=torch.float32):
    """Deterministic values in [-1, 1] from a formula."""
    n = math.prod(shape) if shape else 1
    i = torch.arange(n, dtype=torch.float64)
    values = torch.sin(i * 0.7310585 + k) * torch.cos(i * 0.3 + 2 * k)
    return values.reshape(shape).to(dtype)


def pos(*shape, k=0.0, dtype=torch.float32):
    return det(*shape, k=k, dtype=torch.float64).abs().add(0.1).to(dtype)


def spd(n, k=0.0, dtype=torch.float64):
    a = det(n, n, k=k, dtype=dtype)
    return a @ a.T + n * torch.eye(n, dtype=dtype)


# Elementwise -----------------------------------------------------------------

UNARY = [
    "abs", "neg", "exp", "exp2", "expm1", "sin", "cos", "tan", "tanh", "sinh", "cosh",
    "asin", "acos", "atan", "asinh", "atanh", "sigmoid", "erf", "erfc", "erfinv",
    "floor", "ceil", "round", "trunc", "frac", "sign", "square", "reciprocal",
    "deg2rad", "rad2deg", "sinc", "logit",
]
POSITIVE = ["log", "log2", "log10", "log1p", "sqrt", "rsqrt", "lgamma", "digamma", "acosh_shift"]

for _name in UNARY:
    for _dtype in (torch.float32, torch.float64):
        def _unary(name=_name, dtype=_dtype):
            x = det(4, 37, dtype=dtype) * (0.99 if name in ("asin", "acos", "atanh", "erfinv") else 3.0)
            if name == "logit":
                x = x.abs() * 0.9 + 0.05
            return getattr(torch, name)(x)
        CASES[f"unary.{_name}.{str(_dtype)[6:]}"] = _unary

for _name in POSITIVE:
    for _dtype in (torch.float32, torch.float64):
        def _positive(name=_name, dtype=_dtype):
            x = pos(4, 37, dtype=dtype) * 5
            if name == "acosh_shift":
                return torch.acosh(x + 1)
            return getattr(torch, name)(x)
        CASES[f"unary.{_name}.{str(_dtype)[6:]}"] = _positive

BINARY = ["add", "sub", "mul", "div", "maximum", "minimum", "atan2", "hypot", "fmod",
          "remainder", "copysign", "nextafter", "xlogy", "logaddexp", "pow_pos"]
for _name in BINARY:
    for _dtype in (torch.float32, torch.float64):
        def _binary(name=_name, dtype=_dtype):
            a = det(3, 41, k=0.1, dtype=dtype) * 4
            b = det(3, 41, k=1.7, dtype=dtype) * 4 + (5 if name in ("div", "fmod", "remainder") else 0)
            if name == "pow_pos":
                return torch.pow(a.abs() + 0.5, b / 2)
            if name == "xlogy":
                return torch.xlogy(a, b.abs() + 0.1)
            return getattr(torch, name)(a, b)
        CASES[f"binary.{_name}.{str(_dtype)[6:]}"] = _binary


@case("broadcast.add_mul")
def _():
    return det(4, 1, 5) * det(3, 1, k=1) + det(5, k=2)


@case("elementwise.clamp_where_lerp_addcmul")
def _():
    a, b, c = det(64), det(64, k=1), det(64, k=2)
    return (a.clamp(-0.3, 0.4), torch.where(a > b, a, c), torch.lerp(a, b, 0.3),
            torch.addcmul(a, b, c, value=0.5), torch.addcdiv(a, b, pos(64), value=2.0))


@case("elementwise.int_ops")
def _():
    a = torch.arange(-20, 20, dtype=torch.int64)
    b = torch.arange(1, 41, dtype=torch.int64)
    return (a + b, a * b, torch.div(a, b, rounding_mode="floor"), a % b, a // 3,
            a & b, a | b, a ^ b, a << 2, b >> 1, torch.bitwise_not(a), a.abs())


@case("elementwise.int32_and_uint8")
def _():
    a = torch.arange(0, 200, 7, dtype=torch.uint8)
    b = torch.arange(-50, 50, 3, dtype=torch.int32)
    return (a + 100, a * 3, b * b, b.sum(), a.max())


@case("elementwise.comparisons")
def _():
    a, b = det(80), det(80, k=0.5)
    return (a > b, a <= b, torch.isclose(a, b, atol=0.2), torch.eq(a.round(), b.round()),
            torch.isfinite(a / (b - b)), torch.isnan(torch.sqrt(a)))


@case("elementwise.nan_inf")
def _():
    x = torch.tensor([0.0, -0.0, 1.0, -1.0, float("inf"), float("-inf"), float("nan")])
    return (torch.nan_to_num(x / x), torch.nan_to_num(1 / x), torch.sign(x).nan_to_num(7))


@case("half.ops")
def _():
    a, b = det(8, 16, dtype=torch.float16), det(16, 8, k=1, dtype=torch.float16)
    return (a + 1, a * b.T, a @ b, a.sum(), torch.relu(a), a.exp())


@case("bfloat16.ops")
def _():
    a, b = det(8, 16, dtype=torch.bfloat16), det(16, 8, k=1, dtype=torch.bfloat16)
    return (a + 1, a @ b, a.sum(), torch.softmax(a.float(), -1).to(torch.bfloat16))


@case("complex.ops")
def _():
    re, im = det(32), det(32, k=1)
    z = torch.complex(re, im)
    w = torch.polar(pos(32), det(32, k=2))
    return (z * w, z / (w + 1), z.abs(), z.angle(), torch.exp(z), z.conj() * z,
            torch.view_as_real(z.sqrt()), (z @ w), torch.vdot(z, w))


# Reductions and scans --------------------------------------------------------

for _dtype in (torch.float32, torch.float64):
    @case(f"reduce.basic.{str(_dtype)[6:]}")
    def _(dtype=_dtype):
        x = det(6, 50, dtype=dtype) * 3
        return (x.sum(), x.sum(1), x.mean(0), x.prod(1), x.std(1), x.var(0, unbiased=False),
                x.amax(1), x.amin(0), x.argmax(1), x.argmin(0), x.norm(), x.norm(dim=1, p=1),
                x.logsumexp(1), x.abs().sqrt().sum(), torch.linalg.vector_norm(x, ord=3))


@case("reduce.large_sum_and_mean")
def _():
    x = det(1000, 333)
    return (x.sum(), x.mean(), x.sum(0)[:20], x.double().sum(), x.std())


@case("reduce.scans")
def _():
    x = det(5, 30) * 2
    return (x.cumsum(1), x.cumprod(1)[:, :8], torch.cummax(x, 1).values, torch.cummin(x, 0).values,
            torch.logcumsumexp(x, 1), x.diff(dim=1))


@case("reduce.nan_ops")
def _():
    x = det(4, 10)
    x[1, 3] = float("nan")
    x[2, 7] = float("nan")
    return (x.nansum(1), x.nanmean(0), torch.nanmedian(x[0]), x.isnan().sum())


@case("reduce.median_mode_quantile")
def _():
    x = (det(5, 31) * 10).round()
    return (x.median(1).values, torch.mode(x, 1).values, torch.quantile(x.float(), 0.3, dim=1),
            torch.kthvalue(x, 4, dim=1).values)


@case("reduce.sort_topk_unique")
def _():
    x = (det(4, 40) * 7).round()
    s = torch.sort(x, dim=1, stable=True)
    return (s.values, s.indices, x.topk(5, dim=1).values, torch.unique(x),
            torch.unique(x, return_counts=True)[1], torch.argsort(x[0], stable=True),
            torch.unique_consecutive(torch.tensor([1, 1, 2, 2, 2, 3, 1, 1])))


@case("reduce.any_all_count")
def _():
    x = det(6, 20)
    return ((x > 0).any(1), (x > -0.9).all(0), torch.count_nonzero(x > 0.5, dim=1), torch.nonzero(x > 0.95))


@case("reduce.bincount_histc_searchsorted")
def _():
    x = (pos(200) * 9).long()
    edges = torch.linspace(-1, 1, 11)
    return (torch.bincount(x), torch.histc(det(200), bins=8, min=-1, max=1),
            torch.searchsorted(edges, det(30)), torch.bucketize(det(30), edges))


# Shapes and indexing ---------------------------------------------------------

@case("shape.views_and_copies")
def _():
    x = det(2, 3, 4, 5)
    return (x.permute(3, 1, 0, 2).contiguous(), x.transpose(1, 3).reshape(-1)[:50], x.flip(1, 3),
            x.roll(2, dims=3), x.flatten(1), x.unsqueeze(0).squeeze(), x.narrow(3, 1, 3), x[..., ::2],
            x.movedim(0, -1), x.unfold(3, 2, 1)[0, 0])


@case("shape.cat_stack_split")
def _():
    a, b = det(3, 4), det(3, 4, k=1)
    parts = torch.cat([a, b], 1).split([3, 5], dim=1)
    return (torch.stack([a, b], 2), torch.cat([a, b]), parts[1], torch.chunk(a, 2, 1)[1],
            torch.hstack([a, b]), torch.vstack([a, b]), torch.dstack([a, b]), torch.tril(a), torch.triu(a, 1),
            torch.kron(a[:2, :2], b[:2, :2]), torch.block_diag(a[:2, :2], b[:1, :3]))


@case("index.gather_scatter")
def _():
    x = det(5, 8)
    idx = (pos(5, 3) * 7).long() % 8
    src = det(5, 3, k=3)
    out = torch.zeros(5, 8).scatter_add(1, idx, src)
    return (x.gather(1, idx), out, torch.zeros(5, 8).scatter(1, idx, src),
            x.index_select(1, torch.tensor([0, 2, 7])), torch.zeros(8).index_add(0, idx[0], src[0]),
            x.masked_fill(x > 0.3, -1.0), x[x > 0], x.take(torch.tensor([0, 9, 39])),
            torch.zeros(3, 4).index_put((torch.tensor([0, 2]), torch.tensor([1, 3])), torch.tensor([5.0, 6.0])),
            x.scatter_reduce(1, idx, src, reduce="amax", include_self=False))


@case("index.advanced")
def _():
    x = torch.arange(60.0).reshape(3, 4, 5)
    return (x[[0, 2], :, [1, 3]], x[:, torch.tensor([True, False, True, True])], x[1:, -2:, ::3])


@case("create.factories")
def _():
    return (torch.arange(0, 1, 0.1), torch.linspace(-2, 2, 9), torch.logspace(0, 2, 5), torch.eye(3, 4),
            torch.full((2, 3), 7.5), torch.zeros_like(det(2, 2)), torch.tril_indices(4, 4),
            torch.meshgrid(torch.arange(3.0), torch.arange(2.0), indexing="ij")[1],
            torch.cartesian_prod(torch.arange(3), torch.arange(2)))


# Linear algebra --------------------------------------------------------------

for _dtype in (torch.float32, torch.float64):
    @case(f"matmul.shapes.{str(_dtype)[6:]}")
    def _(dtype=_dtype):
        a, b = det(7, 5, dtype=dtype), det(5, 3, k=1, dtype=dtype)
        v, w = det(5, k=2, dtype=dtype), det(5, k=3, dtype=dtype)
        batch, batch2 = det(4, 7, 5, dtype=dtype), det(4, 5, 3, k=1, dtype=dtype)
        return (a @ b, a @ v, v @ w, torch.dot(v, w), torch.mv(a, v), torch.outer(v, w), batch @ batch2,
                torch.bmm(batch, batch2), batch @ b, torch.addmm(det(7, 3, dtype=dtype), a, b, beta=0.5, alpha=2),
                torch.addmv(det(7, dtype=dtype), a, v), torch.addr(det(5, 5, dtype=dtype), v, w),
                torch.baddbmm(det(4, 7, 3, dtype=dtype), batch, batch2), a.T @ a, torch.inner(v, w),
                torch.tensordot(batch, batch2, dims=([2], [1]))[0, 0], torch.chain_matmul(a, b, det(3, 2, dtype=dtype)))


@case("matmul.large_and_strided")
def _():
    a, b = det(128, 96), det(96, 160, k=1)
    return ((a @ b)[::7, ::9], (a.T.contiguous().T @ b)[:5, :5], (a[:, ::2] @ b[::2, :])[:6, :6],
            (a @ b.T[:, :96].T)[:4, :4])


@case("matmul.int_and_bool")
def _():
    a = torch.arange(12).reshape(3, 4)
    b = torch.arange(8).reshape(4, 2)
    return (a @ b, torch.dot(torch.arange(5), torch.arange(5)), (a > 4).float() @ (b > 2).float())


@case("matmul.einsum")
def _():
    a, b, c = det(3, 4), det(4, 5, k=1), det(2, 3, 4, k=2)
    return (torch.einsum("ij,jk->ik", a, b), torch.einsum("bij,jk->bik", c, b), torch.einsum("ii->", det(4, 4)),
            torch.einsum("bij->bji", c), torch.einsum("i,j->ij", det(3), det(4)))


@case("linalg.solve_inverse_det")
def _():
    a = spd(6)
    b = det(6, 2, k=1, dtype=torch.float64)
    sign, logdet = torch.linalg.slogdet(a)
    return (torch.linalg.solve(a, b), torch.linalg.inv(a), torch.linalg.det(a), sign, logdet,
            torch.linalg.solve(a.float(), b.float()), torch.linalg.inv(a.float()), torch.linalg.det(det(4, 4) + 2 * torch.eye(4)))


@case("linalg.cholesky_lu_triangular")
def _():
    a = spd(5)
    l = torch.linalg.cholesky(a)
    p, low, up = torch.linalg.lu(det(5, 5, dtype=torch.float64) + 3 * torch.eye(5, dtype=torch.float64))
    b = det(5, 2, k=4, dtype=torch.float64)
    return (l, low @ up - p.T @ (det(5, 5, dtype=torch.float64) + 3 * torch.eye(5, dtype=torch.float64)),
            torch.linalg.solve_triangular(l, b, upper=False), torch.cholesky_solve(b, l), torch.cholesky_inverse(l))


@case("linalg.qr_svd_invariants")
def _():
    a = det(6, 4, dtype=torch.float64)
    q, r = torch.linalg.qr(a)
    u, s, vh = torch.linalg.svd(a, full_matrices=False)
    s32 = torch.linalg.svdvals(a.float())
    return (q @ r - a, r.diagonal().abs(), q.T @ q, s, (u * s) @ vh - a, s32, torch.linalg.matrix_rank(a),
            torch.linalg.pinv(a) @ a, torch.linalg.matrix_norm(a, ord="nuc"), torch.linalg.cond(a))


@case("linalg.eigh_eig")
def _():
    a = spd(5)
    values, vectors = torch.linalg.eigh(a)
    general = det(4, 4, dtype=torch.float64) + 2 * torch.eye(4, dtype=torch.float64)
    ev = torch.linalg.eigvals(general)
    ev = ev[torch.argsort(ev.real * 1000 + ev.imag)]
    return (values, vectors @ torch.diag(values) @ vectors.T - a, vectors.abs().sum(0), ev,
            torch.linalg.eigvalsh(a.float()))


@case("linalg.lstsq_matrix_exp_power")
def _():
    a, b = det(8, 3, dtype=torch.float64), det(8, 2, k=1, dtype=torch.float64)
    return (torch.linalg.lstsq(a, b).solution, torch.linalg.matrix_exp(det(4, 4, dtype=torch.float64)),
            torch.linalg.matrix_power(det(3, 3, dtype=torch.float64), 5), torch.linalg.cross(det(4, 3), det(4, 3, k=1)),
            torch.linalg.householder_product(*torch.geqrf(det(5, 3, dtype=torch.float64))))


# Neural network functional ---------------------------------------------------

@case("nn.activations")
def _():
    x = det(4, 50) * 4
    return (F.relu(x), F.gelu(x), F.gelu(x, approximate="tanh"), F.silu(x), F.mish(x), F.elu(x), F.selu(x),
            F.celu(x), F.leaky_relu(x, 0.1), F.softplus(x), F.softsign(x), F.hardtanh(x), F.hardswish(x),
            F.hardsigmoid(x), F.logsigmoid(x), F.tanhshrink(x), F.prelu(x, torch.tensor([0.25])), F.glu(x, -1),
            F.relu6(x), F.rrelu(x, training=False), F.threshold(x, 0.5, -2.0))


@case("nn.softmax_family")
def _():
    x = det(5, 33) * 6
    return (F.softmax(x, 1), F.log_softmax(x, 1), F.softmin(x, 0), F.gumbel_softmax(x, hard=False, tau=1.0).shape,
            F.normalize(x, dim=1), torch.softmax(x.double(), 0))


@case("nn.linear_bilinear_embedding")
def _():
    torch.manual_seed(0)
    x = det(6, 5)
    w, b = det(4, 5, k=1), det(4, k=2)
    table = det(10, 3, k=3)
    return (F.linear(x, w, b), F.bilinear(x, det(6, 2, k=4), det(3, 5, 2, k=5), det(3, k=6)),
            F.embedding(torch.tensor([[1, 2], [9, 0]]), table),
            F.embedding_bag(torch.tensor([1, 2, 4, 5, 4, 3]), table, torch.tensor([0, 3]), mode="mean"),
            F.one_hot(torch.tensor([0, 3, 1]), 5))


@case("nn.convolutions")
def _():
    x1, x2, x3 = det(2, 3, 20), det(2, 3, 12, 11), det(1, 2, 6, 7, 5)
    return (F.conv1d(x1, det(4, 3, 5, k=1), det(4, k=2), padding=2, stride=2),
            F.conv2d(x2, det(5, 3, 3, 3, k=1), det(5, k=2), padding=1),
            F.conv2d(x2, det(6, 1, 3, 3, k=3), groups=3, dilation=2),
            F.conv2d(x2, det(4, 3, 5, 5, k=4), stride=(2, 3), padding="valid"),
            F.conv3d(x3, det(3, 2, 3, 3, 3, k=5), padding=1),
            F.conv_transpose1d(x1, det(3, 2, 4, k=6), stride=2),
            F.conv_transpose2d(x2, det(3, 4, 3, 3, k=7), stride=2, padding=1, output_padding=1),
            F.conv2d(x2.to(memory_format=torch.channels_last), det(5, 3, 3, 3, k=1)),
            F.conv2d(x2.double(), det(5, 3, 3, 3, k=1, dtype=torch.float64)))


@case("nn.pooling")
def _():
    x = det(2, 3, 13, 12) * 3
    out, idx = F.max_pool2d(x, 3, stride=2, padding=1, return_indices=True)
    return (out, idx, F.avg_pool2d(x, 2, ceil_mode=True), F.adaptive_avg_pool2d(x, (5, 4)),
            F.adaptive_max_pool2d(x, 3), F.max_unpool2d(out, idx, 3, stride=2, padding=1, output_size=x.shape[-2:]),
            F.lp_pool2d(x.abs(), 2, 3), F.max_pool1d(x[:, :, 0], 3), F.avg_pool3d(x.unsqueeze(1), 2),
            F.fractional_max_pool2d(x, 2, output_size=(5, 5), _random_samples=torch.full((2, 3, 2), 0.5)))


@case("nn.normalization")
def _():
    x = det(4, 6, 5, 5) * 3 + 1
    rm, rv = torch.zeros(6), torch.ones(6)
    train = F.batch_norm(x, rm, rv, det(6, k=1), det(6, k=2), training=True, momentum=0.2)
    return (train, rm, rv, F.batch_norm(x, rm, rv, training=False), F.layer_norm(x, (5, 5), det(5, 5), det(5, 5, k=1)),
            F.group_norm(x, 3, det(6), det(6, k=1)), F.instance_norm(x), F.local_response_norm(x, 3),
            F.rms_norm(x, (5,), det(5, k=4)))


@case("nn.dropout_eval_and_shapes")
def _():
    x = det(3, 40)
    torch.manual_seed(0)
    dropped = F.dropout(x, 0.5, training=True)
    return (F.dropout(x, 0.5, training=False), ((dropped == 0) | torch.isclose(dropped, 2 * x)).all(),
            F.alpha_dropout(x, 0.3, training=False))


@case("nn.losses")
def _():
    logits, target = det(6, 5) * 3, torch.tensor([0, 4, 2, 2, 1, 3])
    a, b = det(6, 5, k=1), det(6, 5, k=2)
    p = torch.sigmoid(a)
    return (F.cross_entropy(logits, target), F.cross_entropy(logits, target, label_smoothing=0.1),
            F.cross_entropy(logits, target, reduction="none", weight=pos(5)), F.nll_loss(F.log_softmax(logits, 1), target),
            F.mse_loss(a, b), F.l1_loss(a, b), F.smooth_l1_loss(a, b, beta=0.3), F.huber_loss(a, b, delta=0.4),
            F.binary_cross_entropy(p, (b > 0).float()), F.binary_cross_entropy_with_logits(a, (b > 0).float(), pos_weight=pos(5)),
            F.kl_div(F.log_softmax(a, 1), F.softmax(b, 1), reduction="batchmean"), F.cosine_embedding_loss(a, b, torch.tensor([1, -1, 1, 1, -1, 1])),
            F.margin_ranking_loss(a[:, 0], b[:, 0], torch.ones(6)), F.hinge_embedding_loss(a, torch.ones(6, 5)),
            F.soft_margin_loss(a, torch.sign(b)), F.multilabel_soft_margin_loss(a, (b > 0).float()),
            F.poisson_nll_loss(a, pos(6, 5)), F.gaussian_nll_loss(a, b, pos(6, 5)),
            F.triplet_margin_loss(a, b, det(6, 5, k=3)), F.multi_margin_loss(logits, target),
            F.ctc_loss(F.log_softmax(det(12, 2, 6), 2), torch.tensor([[1, 2, 2], [3, 4, 0]]), torch.tensor([12, 12]), torch.tensor([3, 2])))


@case("nn.distances")
def _():
    a, b = det(5, 4), det(7, 4, k=1)
    return (torch.cdist(a, b), torch.cdist(a, b, p=1), F.pairwise_distance(a, b[:5]), F.cosine_similarity(a, b[:5]),
            torch.pdist(a), torch.cdist(a.double(), b.double()))


@case("nn.interpolate_pad_unfold")
def _():
    x = det(1, 2, 5, 6)
    return (F.interpolate(x, scale_factor=2, mode="nearest"), F.interpolate(x, size=(7, 9), mode="bilinear", align_corners=False),
            F.interpolate(x, size=(4, 4), mode="bicubic", align_corners=True), F.interpolate(x, scale_factor=0.5, mode="area"),
            F.interpolate(det(1, 2, 9), size=13, mode="linear"), F.pad(x, (1, 2, 0, 1)), F.pad(x, (2, 2, 1, 1), mode="reflect"),
            F.pad(x, (1, 1, 1, 1), mode="replicate"), F.pad(x, (1, 1, 1, 1), mode="circular"), F.unfold(x, 2, stride=2),
            F.fold(F.unfold(x, 2), (5, 6), 2), F.pixel_shuffle(det(1, 8, 3, 3), 2), F.pixel_unshuffle(det(1, 2, 4, 6), 2),
            F.grid_sample(x, det(1, 3, 4, 2, k=1) * 0.9, align_corners=False), F.affine_grid(det(1, 2, 3) + torch.tensor([[[1.0, 0, 0], [0, 1, 0]]]), (1, 2, 3, 4), align_corners=False))


@case("nn.attention")
def _():
    q, k, v = det(2, 3, 6, 8), det(2, 3, 6, 8, k=1), det(2, 3, 6, 8, k=2)
    mask = torch.tril(torch.ones(6, 6, dtype=torch.bool))
    mha = nn.MultiheadAttention(8, 2, batch_first=True)
    with torch.no_grad():
        for i, p in enumerate(mha.parameters()):
            p.copy_(det(*p.shape, k=i))
        out, weights = mha(q[:, 0], k[:, 0], v[:, 0])
    return (F.scaled_dot_product_attention(q, k, v), F.scaled_dot_product_attention(q, k, v, is_causal=True),
            F.scaled_dot_product_attention(q, k, v, attn_mask=mask), F.scaled_dot_product_attention(q, k, v, scale=0.3),
            out, weights)


def _load(module):
    with torch.no_grad():
        for i, p in enumerate(module.parameters()):
            p.copy_(det(*p.shape, k=i) * 0.5)
    return module


@case("nn.recurrent")
def _():
    x = det(2, 7, 4)
    with torch.no_grad():
        lstm, gru, rnn = _load(nn.LSTM(4, 5, num_layers=2, batch_first=True, bidirectional=True)), _load(nn.GRU(4, 3, batch_first=True)), _load(nn.RNN(4, 3, nonlinearity="relu"))
        out, (h, c) = lstm(x)
        g, gh = gru(x)
        packed = nn.utils.rnn.pack_padded_sequence(x, torch.tensor([7, 4]), batch_first=True)
        packed_out, _ = gru(packed)
        unpacked, lengths = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        cell = _load(nn.LSTMCell(4, 3))(x[:, 0])
        return (out, h, c, g, gh, rnn(x.transpose(0, 1))[0], unpacked, lengths, cell[0], _load(nn.GRUCell(4, 3))(x[:, 1]))


@case("nn.transformer_layer")
def _():
    torch.manual_seed(0)
    layer = _load(nn.TransformerEncoderLayer(8, 2, dim_feedforward=16, dropout=0.0, batch_first=True))
    decoder = _load(nn.TransformerDecoderLayer(8, 2, dim_feedforward=16, dropout=0.0, batch_first=True))
    layer.eval(), decoder.eval()
    with torch.no_grad():
        x, memory = det(2, 5, 8), det(2, 4, 8, k=1)
        return (layer(x), layer(x, src_mask=nn.Transformer.generate_square_subsequent_mask(5)), decoder(x, memory))


@case("nn.modules_forward")
def _():
    torch.manual_seed(0)
    model = _load(nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4), nn.ReLU(), nn.MaxPool2d(2),
                                nn.Flatten(), nn.Linear(4 * 4 * 4, 6), nn.LayerNorm(6), nn.GELU(), nn.Linear(6, 3)))
    model.eval()
    with torch.no_grad():
        return model(det(2, 3, 8, 8))


# FFT and signal --------------------------------------------------------------

@case("fft.transforms")
def _():
    x = det(4, 30)
    z = torch.complex(x, det(4, 30, k=1))
    return (torch.fft.fft(x), torch.fft.ifft(z), torch.fft.rfft(x), torch.fft.irfft(torch.fft.rfft(x), n=30),
            torch.fft.fft2(det(6, 8)), torch.fft.rfftn(det(3, 4, 6)), torch.fft.fftshift(torch.fft.fftfreq(8)),
            torch.fft.hfft(z[0, :16]), torch.fft.fft(x.double(), norm="ortho"), torch.fft.fft(det(1, 97)))


@case("fft.stft")
def _():
    signal = det(400)
    spec = torch.stft(signal, 64, hop_length=16, window=torch.hann_window(64), return_complex=True)
    return (spec.abs(), torch.istft(spec, 64, hop_length=16, window=torch.hann_window(64), length=400),
            torch.hann_window(10), torch.hamming_window(7), torch.blackman_window(6), torch.kaiser_window(5))


# Special functions -----------------------------------------------------------

@case("special.functions")
def _():
    x = pos(40) * 4
    s = torch.special
    return (s.gammaln(x), s.digamma(x), s.polygamma(1, x), s.erfcx(x), s.i0(x), s.i0e(x), s.i1(x), s.i1e(x),
            s.ndtr(x - 2), s.ndtri(torch.sigmoid(x - 2)), s.log_ndtr(x - 3), s.xlog1py(x, x), s.zeta(x + 1, x),
            s.entr(x / 5), s.sinc(x - 2), s.bessel_j0(x), s.bessel_j1(x), s.multigammaln(x + 2, 3),
            s.logsumexp(det(4, 6), 1), s.log1p(x), s.expit(x - 2), s.gammainc(x, x / 2), s.gammaincc(x, x / 2))


# Random numbers with a seed --------------------------------------------------

@case("random.seeded")
def _():
    torch.manual_seed(1234)
    a = torch.rand(20)
    b = torch.randn(40)
    c = torch.randint(0, 100, (20,))
    d = torch.randperm(15)
    g = torch.Generator().manual_seed(7)
    e = torch.multinomial(torch.tensor([0.1, 0.5, 0.4]), 10, replacement=True, generator=g)
    f = torch.normal(0.0, 1.0, (10,), generator=g)
    return (a, b, c, d, e, f, torch.bernoulli(torch.full((16,), 0.5), generator=g))


@case("random.statistics")
def _():
    torch.manual_seed(0)
    x = torch.randn(200000)
    u = torch.rand(200000)
    return (x.mean().abs() < 0.01, (x.std() - 1).abs() < 0.01, (u.mean() - 0.5).abs() < 0.01,
            u.min() >= 0, u.max() < 1, torch.distributions.Normal(0.0, 2.0).log_prob(torch.tensor([0.5, -1.0])))


# Autograd --------------------------------------------------------------------

@case("autograd.composite_grads")
def _():
    x = det(4, 6, dtype=torch.float64).requires_grad_()
    w = det(6, 3, k=1, dtype=torch.float64).requires_grad_()
    y = (torch.tanh(x @ w) ** 2).sum() + torch.logsumexp(x, 1).sum() + F.softplus(x).mean() + x.norm()
    gx, gw = torch.autograd.grad(y, (x, w), create_graph=True)
    ggx = torch.autograd.grad(gx.sum(), x)[0]
    return (gx, gw, ggx)


GRADCHECK = {
    "matmul": lambda a, b: a @ b,
    "conv2d": lambda a, b: F.conv2d(a.reshape(1, 1, 4, 4), b[:2, :2].reshape(1, 1, 2, 2)),
    "softmax_log": lambda a, b: F.log_softmax(a @ b, 1),
    "norms": lambda a, b: torch.linalg.matrix_norm(a @ b) + torch.linalg.vector_norm(a, 3),
    "solve": lambda a, b: torch.linalg.solve(a @ a.T + 4 * torch.eye(4, dtype=a.dtype), b),
    "cholesky": lambda a, b: torch.linalg.cholesky(a @ a.T + 4 * torch.eye(4, dtype=a.dtype)),
    "det": lambda a, b: torch.linalg.det(a + 3 * torch.eye(4, dtype=a.dtype)),
    "eigh": lambda a, b: torch.linalg.eigvalsh(a + a.T),
    "svdvals": lambda a, b: torch.linalg.svdvals(a + 2 * torch.eye(4, dtype=a.dtype)),
    "layer_norm": lambda a, b: F.layer_norm(a, (4,)),
    "attention": lambda a, b: F.scaled_dot_product_attention(a.unsqueeze(0), b.T.unsqueeze(0), b.T.unsqueeze(0)),
    "cumprod": lambda a, b: torch.cumprod(a + 2, 1),
    "pool": lambda a, b: F.max_pool2d((a @ b).reshape(1, 1, 4, 4), 2) + F.avg_pool2d((a @ b).reshape(1, 1, 4, 4), 2),
    "interpolate": lambda a, b: F.interpolate(a.reshape(1, 1, 4, 4), size=(6, 5), mode="bilinear", align_corners=False),
    "fft": lambda a, b: torch.fft.rfft(a @ b).abs(),
    "gelu_erf": lambda a, b: F.gelu(a) * torch.erf(b.T),
    "lstm_cell": lambda a, b: torch.lstm_cell(a, (b.T, b.T), torch.ones(16, 4, dtype=a.dtype) * 0.1, torch.ones(16, 4, dtype=a.dtype) * 0.2)[0],
}

for _name, _fn in GRADCHECK.items():
    def _gradcheck(fn=_fn):
        a = det(4, 4, dtype=torch.float64).requires_grad_()
        b = det(4, 4, k=1, dtype=torch.float64).requires_grad_()
        return torch.autograd.gradcheck(fn, (a, b), eps=1e-6, atol=1e-5, raise_exception=False)
    CASES[f"gradcheck.{_name}"] = _gradcheck


@case("autograd.training_step")
def _():
    model = _load(nn.Sequential(nn.Linear(5, 8), nn.Tanh(), nn.Linear(8, 2)))
    results = []
    for maker in (lambda p: torch.optim.SGD(p, lr=0.1, momentum=0.9, nesterov=True), lambda p: torch.optim.Adam(p, lr=0.01),
                  lambda p: torch.optim.AdamW(p, lr=0.01, weight_decay=0.1), lambda p: torch.optim.RMSprop(p, lr=0.01),
                  lambda p: torch.optim.Adagrad(p, lr=0.1), lambda p: torch.optim.LBFGS(p, lr=0.5, max_iter=5)):
        net = _load(nn.Sequential(nn.Linear(5, 8), nn.Tanh(), nn.Linear(8, 2)))
        opt = maker(net.parameters())
        x, y = det(16, 5), det(16, 2, k=1)
        for _ in range(5):
            def closure():
                opt.zero_grad()
                loss = F.mse_loss(net(x), y)
                loss.backward()
                return loss
            opt.step(closure)
        results.append(F.mse_loss(net(x), y).detach())
    return torch.stack(results)


@case("autograd.functional_api")
def _():
    f = lambda x: (x.sin() * x).sum()
    x = det(5, dtype=torch.float64)
    return (torch.autograd.functional.jacobian(lambda v: v.exp() * v[0], x), torch.autograd.functional.hessian(f, x),
            torch.func.grad(f)(x), torch.func.vmap(torch.func.grad(lambda v: (v ** 3).sum()))(det(3, 4, dtype=torch.float64)),
            torch.func.jacrev(torch.sin)(x), torch.func.jacfwd(torch.cos)(x), torch.func.hessian(f)(x),
            torch.func.vmap(torch.dot)(det(3, 4), det(3, 4, k=1)))


@case("autograd.checkpoint_and_hooks")
def _():
    from torch.utils.checkpoint import checkpoint
    lin = _load(nn.Linear(4, 4))
    x = det(3, 4).requires_grad_()
    seen = []
    x.register_hook(lambda g: seen.append(g.sum()))
    y = checkpoint(lambda t: torch.tanh(lin(t)), x, use_reentrant=False).sum()
    y.backward()
    return (x.grad, torch.stack(seen))


# Sparse, quantised and other paths --------------------------------------------

@case("sparse.basic")
def _():
    dense = det(5, 6) * (det(5, 6, k=1) > 0.3)
    s = dense.to_sparse()
    return (s.to_dense(), torch.sparse.mm(s, det(6, 3)), (s + s).to_dense(), s._nnz(), dense.to_sparse_csr().to_dense(),
            torch.sparse.sum(s, dim=1).to_dense())


@case("quantized.basic")
def _():
    x = det(4, 8) * 3
    q = torch.quantize_per_tensor(x, scale=0.05, zero_point=10, dtype=torch.quint8)
    qc = torch.quantize_per_channel(x, torch.full((4,), 0.04), torch.zeros(4, dtype=torch.long), 0, torch.qint8)
    return (q.dequantize(), q.int_repr(), qc.dequantize(), torch.tensor("qnnpack" in torch.backends.quantized.supported_engines))


@case("dtype.conversions")
def _():
    x = det(50) * 300
    return (x.to(torch.int32), x.to(torch.uint8), x.to(torch.int8), x.to(torch.float16), x.to(torch.bfloat16),
            x.to(torch.float64), x.to(torch.bool), x.round().to(torch.int64).to(torch.float32),
            torch.tensor([1.5, 2.5, -0.5]).round(), x.half().float() - x)


@case("numpy.interop")
def _():
    import numpy as np
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    t = torch.from_numpy(a)
    t[0, 0] = 99
    back = (t * 2).numpy()
    return (torch.tensor(a[0, 0]), torch.from_numpy(back), torch.as_tensor(np.array([1, 2, 3], dtype=np.int16)),
            torch.from_numpy(np.ones((2, 3), dtype=np.float64)) @ torch.ones(3, 2, dtype=torch.float64),
            torch.tensor(np.float32(3.5)), torch.from_numpy(a.T.copy()))


@case("serialize.save_load_jit")
def _():
    model = _load(nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2)))
    buffer = io.BytesIO()
    torch.save({"model": model.state_dict(), "step": 7, "tensor": det(3)}, buffer)
    buffer.seek(0)
    loaded = torch.load(buffer)
    copy = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))
    copy.load_state_dict(loaded["model"])
    scripted = torch.jit.script(model)
    jit_buffer = io.BytesIO()
    torch.jit.save(scripted, jit_buffer)
    jit_buffer.seek(0)
    reloaded = torch.jit.load(jit_buffer)
    traced = torch.jit.trace(model, det(1, 3))
    x = det(5, 3, k=2)
    with torch.no_grad():
        return (copy(x), reloaded(x), traced(x), torch.tensor(loaded["step"]), loaded["tensor"])


@case("data.loader")
def _():
    from torch.utils.data import DataLoader, TensorDataset, random_split
    data = TensorDataset(torch.arange(20.0).unsqueeze(1), torch.arange(20))
    g = torch.Generator().manual_seed(3)
    train, test = random_split(data, [15, 5], generator=g)
    batches = [y for _, y in DataLoader(data, batch_size=6, shuffle=False)]
    return (torch.cat(batches), torch.tensor(sorted(int(test[i][1]) for i in range(5))).shape, torch.tensor(len(train)))


@case("threads.parallel_ops")
def _():
    before = torch.get_num_threads()
    x = det(512, 512)
    full = (x @ x).sum(), x.exp().sum(), x.sort(1).values.sum()
    torch.set_num_threads(1)
    single = (x @ x).sum(), x.exp().sum(), x.sort(1).values.sum()
    torch.set_num_threads(before)
    return (torch.stack(full), torch.stack(single), torch.tensor(before >= 1))


@case("info.backends")
def _():
    return (torch.tensor(torch.backends.mps.is_available()), torch.tensor(torch.cuda.is_available()),
            torch.tensor(torch.backends.openmp.is_available()), torch.tensor(torch.backends.mkldnn.is_available()))


@case("quantized.dynamic_linear")
def _():
    torch.backends.quantized.engine = "qnnpack"
    model = _load(nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4)))
    q = torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    with torch.no_grad():
        return (q(det(5, 8)), torch.tensor(torch.backends.quantized.engine == "qnnpack"))


@case("quantized.static_conv")
def _():
    torch.backends.quantized.engine = "qnnpack"
    import torch.ao.quantization as tq

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.quant, self.dequant = tq.QuantStub(), tq.DeQuantStub()
            self.conv, self.relu = nn.Conv2d(2, 4, 3), nn.ReLU()

        def forward(self, x):
            return self.dequant(self.relu(self.conv(self.quant(x))))

    net = _load(Net()).eval()
    net.qconfig = tq.get_default_qconfig("qnnpack")
    prepared = tq.prepare(net)
    with torch.no_grad():
        prepared(det(4, 2, 8, 8) * 2)
        converted = tq.convert(prepared)
        return converted(det(2, 2, 8, 8, k=1) * 2)


@case("quantized.functional_ops")
def _():
    torch.backends.quantized.engine = "qnnpack"
    x = torch.quantize_per_tensor(det(2, 3, 6, 6) * 2, 0.02, 128, torch.quint8)
    return (torch.nn.quantized.functional.max_pool2d(x, 2).dequantize(), torch.ops.quantized.add(x, x, 0.04, 128).dequantize(),
            F.relu(x).dequantize(), x.mean().dequantize() if hasattr(x.mean(), "dequantize") else x.dequantize().mean())


@case("compile.eager_backends")
def _():
    model = _load(nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 2)))
    x = det(3, 4)
    torch._dynamo.reset()
    eager = torch.compile(model, backend="eager")(x)
    torch._dynamo.reset()
    aot = torch.compile(model, backend="aot_eager")(x)
    return (eager.detach(), aot.detach())


@case("export.program")
def _():
    model = _load(nn.Sequential(nn.Linear(4, 6), nn.Tanh(), nn.Linear(6, 2))).eval()
    x = det(3, 4)
    program = torch.export.export(model, (x,))
    with torch.no_grad():
        return (program.module()(x), torch.tensor(len(program.graph.nodes)))


@case("amp.autocast_bfloat16")
def _():
    a, b = det(16, 32), det(32, 8, k=1)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = a @ b
        lin = F.linear(a, b.T)
    return (out.float(), torch.tensor(out.dtype == torch.bfloat16), lin.float())


@case("nn.init_seeded")
def _():
    torch.manual_seed(5)
    w = torch.empty(6, 8)
    out = []
    for init in (nn.init.xavier_uniform_, nn.init.xavier_normal_, nn.init.kaiming_uniform_, nn.init.kaiming_normal_,
                 nn.init.orthogonal_, nn.init.trunc_normal_, nn.init.uniform_, nn.init.normal_):
        out.append(init(w.clone()))
    return tuple(out)


@case("nn.utils_parametrize_clip")
def _():
    lin = _load(nn.Linear(5, 4))
    weight_norm = torch.nn.utils.parametrizations.weight_norm(_load(nn.Linear(5, 4)))
    spectral = torch.nn.utils.parametrizations.spectral_norm(_load(nn.Linear(5, 4)))
    orth = torch.nn.utils.parametrizations.orthogonal(_load(nn.Linear(4, 4)))
    loss = (lin(det(3, 5)) ** 2).sum() * 100
    loss.backward()
    total = torch.nn.utils.clip_grad_norm_(lin.parameters(), 1.0)
    vector = torch.nn.utils.parameters_to_vector(lin.parameters())
    x = det(2, 5)
    with torch.no_grad():
        return (total.detach(), lin.weight.grad.norm(), vector.detach(), weight_norm(x), spectral.weight.shape,
                orth.weight @ orth.weight.T)


@case("distributions.log_prob_and_sampling")
def _():
    d = torch.distributions
    x = pos(6)
    torch.manual_seed(11)
    samples = (d.Normal(0.0, 1.0).sample((5,)), d.Categorical(probs=torch.tensor([0.2, 0.3, 0.5])).sample((5,)),
               d.Dirichlet(torch.ones(3)).sample(), d.Gamma(2.0, 1.0).sample((3,)), d.Beta(2.0, 3.0).sample((3,)))
    return (d.Normal(0.5, 2.0).log_prob(x), d.Bernoulli(probs=0.3).log_prob(torch.tensor([0.0, 1.0])),
            d.Categorical(logits=det(4)).entropy(), d.Gamma(2.0, 3.0).log_prob(x), d.Beta(2.0, 3.0).log_prob(x / 2),
            d.MultivariateNormal(torch.zeros(3), spd(3).float()).log_prob(det(4, 3)),
            d.kl_divergence(d.Normal(0.0, 1.0), d.Normal(1.0, 2.0)), d.Poisson(3.0).log_prob(torch.tensor([0.0, 2.0, 5.0])),
            d.Normal(0.0, 1.0).cdf(det(5)), d.Normal(0.0, 1.0).rsample((2,)).shape, *samples)


@case("tensor.misc_methods")
def _():
    x = det(4, 5)
    return (x.clone().fill_diagonal_(9), torch.diag_embed(det(2, 3)), torch.diagonal(det(3, 4, 4), dim1=1, dim2=2),
            torch.trace(det(4, 4)), torch.rot90(x, 1, (0, 1)), torch.vander(det(4), N=3), torch.cov(x),
            torch.corrcoef(x), torch.trapezoid(x, dim=1), torch.cumulative_trapezoid(x, dim=1), torch.logdet(spd(3)),
            torch.renorm(x, 2, 0, 0.5), torch.histogram(det(50), bins=5).hist, torch.bucketize(x, torch.tensor([-0.5, 0.0, 0.5])),
            torch.clamp(x, min=torch.zeros(5)), x.new_ones(2, 2), x.storage_offset(), torch.isin(torch.arange(6), torch.tensor([1, 4])))


# Run -------------------------------------------------------------------------

def encode(value):
    if isinstance(value, torch.Tensor):
        t = value.detach()
        if t.is_sparse or t.layout != torch.strided:
            t = t.to_dense()
        if t.is_quantized:
            t = t.dequantize()
        info = {"dtype": str(t.dtype), "shape": list(t.shape)}
        if t.is_complex():
            t = torch.view_as_real(t.resolve_conj())
        if t.dtype in (torch.float16, torch.bfloat16):
            t = t.float()
        info["values"] = t.reshape(-1).tolist()
        return info
    if isinstance(value, torch.Size):
        return {"size": list(value)}
    if isinstance(value, (tuple, list)):
        return {"items": [encode(item) for item in value]}
    if isinstance(value, (bool, int, float)):
        return {"number": float(value)}
    if isinstance(value, str):
        return {"text": value}
    return {"text": repr(value)}


def main(path):
    results = {"torch": torch.__version__, "cases": {}}
    for name, function in CASES.items():
        try:
            results["cases"][name] = encode(function())
        except Exception as error:
            results["cases"][name] = {"error": f"{type(error).__name__}: {error}"[:400],
                                      "where": traceback.format_exc().splitlines()[-3:]}
    with open(path, "w") as handle:
        json.dump(results, handle)
    failed = [name for name, value in results["cases"].items() if "error" in value]
    print(f"{len(CASES)} cases, {len(failed)} raised")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "conformance.json")
