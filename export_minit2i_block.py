import os, sys, json, traceback, inspect
import numpy as np

OUT = "out"
os.makedirs(OUT, exist_ok=True)
LOG = []

def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)

def save():
    try:
        open(os.path.join(OUT, "export_log.txt"), "w").write("\n".join(LOG))
    except Exception as e:
        print("save log failed:", e)

log("python:", sys.version)

# ---------- 1. 下载 ----------
log("\n=== 1. download model ===")
try:
    from huggingface_hub import snapshot_download
    local = snapshot_download(
        "BiliSakura/MiniT2I-diffusers",
        allow_patterns=["MiniT2I-B-16/transformer/*"],
    )
    log("ok:", local)
except Exception as e:
    log("FAIL:", e); log(traceback.format_exc()); save(); sys.exit(0)

# ---------- 2. 载入 ----------
log("\n=== 2. load model ===")
model = None
try:
    import torch, diffusers
    log("torch", torch.__version__, "| diffusers", diffusers.__version__)
    from diffusers import AutoModel
    model = AutoModel.from_pretrained(
        "BiliSakura/MiniT2I-diffusers",
        subfolder="MiniT2I-B-16/transformer",
        trust_remote_code=True,
    )
    log("loaded:", type(model).__name__)
    log("config:", json.dumps(dict(model.config), indent=1, default=str))
except Exception as e:
    log("FAIL:", e); log(traceback.format_exc())

# ---------- 3. 结构 ----------
net = None
log("\n=== 3. structure ===")
if model is not None:
    try:
        for n, m in model.named_children():
            log("  child %-18s %s (%d params)" % (
                n, type(m).__name__, sum(p.numel() for p in m.parameters())))
        try:
            net = model.model.net
        except Exception:
            for n, m in model.named_modules():
                if hasattr(m, "blocks"):
                    net = m; log("  net found:", n, type(m).__name__); break
        if net is not None:
            log("  net:", type(net).__name__)
            for n, m in net.named_children():
                log("    %-22s %s" % (n, type(m).__name__))
            b0 = net.blocks[0]
            log("  block[0]:", type(b0).__name__)
            for n, m in b0.named_children():
                log("    %-22s %s" % (n, type(m).__name__))
            log("  --- block[0] params ---")
            for n, p in b0.named_parameters():
                log("    %-40s %s" % (n, tuple(p.shape)))
            try:
                log("  --- block.forward source ---")
                log(inspect.getsource(type(b0).forward))
            except Exception as e:
                log("  (no source:", e, ")")
    except Exception as e:
        log("FAIL:", e); log(traceback.format_exc())

# ---------- 4. 卷积改写校验 ----------
log("\n=== 4. conv->linear equivalence ===")
try:
    import torch
    if net is not None and hasattr(net, "img_embedder"):
        pe = net.img_embedder
        for n, m in pe.named_children():
            log("  %-10s %s" % (n, m))
        p1, p2 = getattr(pe, "proj1", None), getattr(pe, "proj2", None)
        if p1 is not None and p2 is not None:
            log("  proj1.weight", tuple(p1.weight.shape), "stride", p1.stride, "pad", p1.padding)
            log("  proj2.weight", tuple(p2.weight.shape), "stride", p2.stride, "pad", p2.padding)
            C, PC, PS = p1.in_channels, p1.out_channels, p1.kernel_size[0]
            G = 32
            x = torch.randn(1, C, G * PS, G * PS)
            with torch.no_grad():
                ref = pe(x)
                pt = x.unfold(2, PS, PS).unfold(3, PS, PS)
                pt = pt.contiguous().permute(0, 2, 3, 1, 4, 5).reshape(1, G * G, C * PS * PS)
                w1 = p1.weight.reshape(PC, C * PS * PS)
                t = pt @ w1.t()
                if p2.bias is not None:
                    t = t + p2.bias
                r = ref
                if r.dim() == 4:
                    r = r.flatten(2).transpose(1, 2)
                log("  ref shape", tuple(r.shape), "rewritten shape", tuple(t.shape))
                log("  max abs err:", float((r - t).abs().max()))
                log("  allclose(atol=1e-5):", bool(torch.allclose(r, t, atol=1e-5, rtol=1e-4)))
    else:
        log("  no img_embedder, skipped")
except Exception as e:
    log("FAIL:", e); log(traceback.format_exc())

# ---------- 5. 提取 block + 前向 + 算子清单 ----------
log("\n=== 5. block export ===")
block = None
if net is not None:
    try:
        block = net.blocks[0].eval()
        log("  block:", type(block).__name__)
    except Exception as e:
        log("FAIL:", e)

if block is not None:
    import torch
    B, LI, LT, H = 1, 1024, 256, 768
    x   = torch.randn(B, LI, H)
    txt = torch.randn(B, LT, H)
    vec = torch.randn(B, H)
    try:
        with torch.no_grad():
            y = block(x, txt, vec)
        if isinstance(y, (tuple, list)):
            for i, t in enumerate(y):
                log("  out[%d] %s" % (i, tuple(t.shape)))
            ox, ot = y[0], (y[1] if len(y) > 1 else None)
        else:
            log("  out", tuple(y.shape)); ox, ot = y, None
        d = {"x": x.numpy(), "txt": txt.numpy(), "vec": vec.numpy(), "out_x": ox.numpy()}
        if ot is not None:
            d["out_txt"] = ot.numpy()
        np.savez(os.path.join(OUT, "block_ref.npz"), **d)
        log("  saved block_ref.npz  out_x[0,0,:5] =", ox[0, 0, :5].tolist())
    except Exception as e:
        log("  forward FAIL:", e); log(traceback.format_exc())

    try:
        ep = torch.export.export(block, (x, txt, vec))
        g = ep.graph_module.graph
        cnt = {}
        for nd in g.nodes:
            if nd.op == "call_function":
                k = str(nd.target)
                cnt[k] = cnt.get(k, 0) + 1
        lines = ["%5d  %s" % (v, k) for k, v in sorted(cnt.items(), key=lambda kv: -kv[1])]
        log("  --- aten ops ---")
        for l in lines:
            log("  " + l)
        open(os.path.join(OUT, "block_ops.txt"), "w").write(
            "MiniT2I DoubleStreamDiTBlock #0 aten ops\n" + "=" * 50 + "\n" +
            "\n".join(lines) + "\n\n---- full graph ----\n" + str(g))
        log("  saved block_ops.txt")
    except Exception as e:
        log("  torch.export FAIL:", e); log(traceback.format_exc())

    # ---------- 6. TFLite ----------
    log("\n=== 6. tflite ===")
    done = False
    for pkg in ("ai_edge_torch", "litert_torch"):
        try:
            m = __import__(pkg)
            log("  using", pkg, getattr(m, "__version__", "?"))
            m.convert(block, (x, txt, vec)).export(os.path.join(OUT, "block.tflite"))
            log("  OK:", os.path.getsize(os.path.join(OUT, "block.tflite")), "bytes")
            done = True
            break
        except Exception as e:
            log("  %s FAIL: %s %s" % (pkg, type(e).__name__, e))
    if not done:
        log("  tflite skipped (block_ops.txt / block_ref.npz still produced)")

log("\n=== files ===")
for f in sorted(os.listdir(OUT)):
    log("  %10d  %s" % (os.path.getsize(os.path.join(OUT, f)), f))
save()
