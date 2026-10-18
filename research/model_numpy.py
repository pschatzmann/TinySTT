"""numpy reference of TinySTT's AcousticModel (src/TinySTT/AcousticModel.h).

Used to produce golden outputs for the C++ tests and to validate exports of
the PyTorch model (research/train.py must agree with this within float
tolerance). Also implements the weight-only int8 quantization the C++ int8
path expects: per output row, symmetric, scale = max|w| / 127.
"""
import numpy as np

CONFIG_KEYS = ["version", "n_mels", "conv_channels", "conv_kernel", "conv_stride", "gru_layers",
               "hidden", "n_classes"]
CONFIG_VERSION = 1

# names of all dense weights (2D after flattening) and their biases
def dense_names(cfg):
    names = [("conv.weight", "conv.bias")]
    for l in range(cfg["gru_layers"]):
        names.append((f"gru{l}.weight_ih", f"gru{l}.bias_ih"))
        names.append((f"gru{l}.weight_hh", f"gru{l}.bias_hh"))
    names.append(("out.weight", "out.bias"))
    return names


def quantize_rows(w2d):
    """float [rows, cols] -> (int8 [rows, cols], float32 scale [rows])."""
    w2d = np.asarray(w2d, dtype=np.float32)
    mx = np.abs(w2d).max(axis=1)
    scale = np.where(mx > 0, mx / 127.0, 1.0).astype(np.float32)
    q = np.clip(np.round(w2d / scale[:, None]), -127, 127).astype(np.int8)
    return q, scale


def config_array(cfg):
    return np.array([CONFIG_VERSION] + [cfg[k] for k in CONFIG_KEYS[1:]], dtype="<u4")


def config_from_array(a):
    if int(a[0]) != CONFIG_VERSION:
        raise ValueError("unsupported model.config version")
    return {k: int(v) for k, v in zip(CONFIG_KEYS, a)}


def add_model(writer, cfg, weights, int8=True, float32=True, mean=None, inv_std=None):
    """Writes model.config, weights (f32 and/or int8+scale) and biases."""
    writer.add("model.config", config_array(cfg))
    for wname, bname in dense_names(cfg):
        w = np.asarray(weights[wname], dtype=np.float32)
        w2d = w.reshape(w.shape[0], -1)
        if float32:
            writer.add(wname, w2d.astype("<f4"))
        if int8:
            q, s = quantize_rows(w2d)
            writer.add(wname + ".q", q)
            writer.add(wname + ".scale", s)
        writer.add(bname, np.asarray(weights[bname], dtype="<f4"))
    if mean is not None:
        writer.add("features.mean", np.asarray(mean, dtype="<f4"))
        writer.add("features.inv_std", np.asarray(inv_std, dtype="<f4"))


def quantize_vector(x):
    """Per-vector symmetric int8 activation quantization exactly like
    DenseWeights::applyQ8 (float32, scale = max|x|/127, round half to even)."""
    x = np.asarray(x, dtype=np.float32)
    mx = np.float32(np.abs(x).max()) if x.size else np.float32(0)
    xs = np.float32(mx / np.float32(127.0)) if mx > 0 else np.float32(1.0)
    inv = np.float32(1.0) / xs
    xq = np.clip(np.rint(x * inv), -127, 127).astype(np.int32)
    return xq, xs


def _dense(weights, name, use_int8, act_quant=False):
    w = np.asarray(weights[name], dtype=np.float32)
    w2d = w.reshape(w.shape[0], -1)
    if use_int8 and act_quant:
        q, s = quantize_rows(w2d)
        q32 = q.astype(np.int64)

        def f(x):
            xq, xs = quantize_vector(x)
            return (q32 @ xq).astype(np.float32) * s * xs
        return f
    if use_int8:
        q, s = quantize_rows(w2d)
        return lambda x: (q.astype(np.float32) @ x) * s
    return lambda x: w2d @ x


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def forward(cfg, weights, features, use_int8=False, act_quant=False):
    """features [T, n_mels] -> log-probs [ceil(T/stride), n_classes] (streaming-equivalent).
    use_int8: int8 weights; act_quant: also int8 activations (AcousticModel Int8Activations)."""
    n, k, s, H = cfg["n_mels"], cfg["conv_kernel"], cfg["conv_stride"], cfg["hidden"]
    conv = _dense(weights, "conv.weight", use_int8, act_quant)
    conv_b = np.asarray(weights["conv.bias"], dtype=np.float32)
    ih = [_dense(weights, f"gru{l}.weight_ih", use_int8, act_quant) for l in range(cfg["gru_layers"])]
    hh = [_dense(weights, f"gru{l}.weight_hh", use_int8, act_quant) for l in range(cfg["gru_layers"])]
    bih = [np.asarray(weights[f"gru{l}.bias_ih"], dtype=np.float32) for l in range(cfg["gru_layers"])]
    bhh = [np.asarray(weights[f"gru{l}.bias_hh"], dtype=np.float32) for l in range(cfg["gru_layers"])]
    out = _dense(weights, "out.weight", use_int8, act_quant)
    out_b = np.asarray(weights["out.bias"], dtype=np.float32)

    T = features.shape[0]
    padded = np.concatenate([np.zeros((k - 1, n), np.float32), features.astype(np.float32)])
    h = [np.zeros(H, np.float32) for _ in range(cfg["gru_layers"])]
    result = []
    for t in range(0, T, s):
        win = padded[t:t + k]          # inputs t-k+1 .. t
        flat = win.T.reshape(-1)       # [n_mels][K] like torch Conv1d weight
        x = np.maximum(conv(flat) + conv_b, 0.0)
        for l in range(cfg["gru_layers"]):
            gi = ih[l](x) + bih[l]
            gh = hh[l](h[l]) + bhh[l]
            r = _sigmoid(gi[:H] + gh[:H])
            z = _sigmoid(gi[H:2 * H] + gh[H:2 * H])
            nn = np.tanh(gi[2 * H:] + r * gh[2 * H:])
            h[l] = (1 - z) * nn + z * h[l]
            x = h[l]
        logits = out(x) + out_b
        m = logits.max()
        result.append(logits - (m + np.log(np.exp(logits - m).sum())))
    return np.array(result, dtype=np.float32)


def random_weights(cfg, seed=0, scale=1.0):
    """Deterministic random weights with roughly unit-variance activations."""
    rng = np.random.default_rng(seed)
    n, C, k, H, L, V = (cfg[x] for x in ["n_mels", "conv_channels", "conv_kernel", "hidden",
                                           "gru_layers", "n_classes"])
    w = {"conv.weight": rng.normal(0, scale / np.sqrt(n * k), (C, n, k)),
         "conv.bias": rng.normal(0, 0.1, C)}
    for l in range(L):
        inp = C if l == 0 else H
        w[f"gru{l}.weight_ih"] = rng.normal(0, scale / np.sqrt(inp), (3 * H, inp))
        w[f"gru{l}.weight_hh"] = rng.normal(0, scale / np.sqrt(H), (3 * H, H))
        w[f"gru{l}.bias_ih"] = rng.normal(0, 0.1, 3 * H)
        w[f"gru{l}.bias_hh"] = rng.normal(0, 0.1, 3 * H)
    w["out.weight"] = rng.normal(0, 2.0 * scale / np.sqrt(H), (V, H))
    w["out.bias"] = rng.normal(0, 0.1, V)
    return {k2: v.astype(np.float32) for k2, v in w.items()}
