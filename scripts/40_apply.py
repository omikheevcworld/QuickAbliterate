"""Apply the chosen refusal direction to the text LM in HF-land.

Projects the unit direction d (residual space, shape [hidden]) out of the
designed writer tensors across SURGERY_LAYERS ("48" | "48,49,50" | "27-60" |
"all"), with projector strength ALPHA (default 1.0). DIRECTION is the .pt to
apply (required).

SURGERY_KINDS selects WHICH residual-writing tensors get the projector per
layer (default "mlp"). All of these write into the residual stream, and the
d(d^T W) orthogonalization is identical in form for each:
    mlp    layer.mlp.down_proj         W [hidden, intermediate]
    o      layer.self_attn.o_proj      W [hidden, qkv-agg]   (full-attn layers)
    gdn    layer.linear_attn.out_proj  W [hidden, ...]       (GatedDeltaNet layers)
    embed  model.embed_tokens.weight   E [vocab, hidden] (reading side:
           E <- E - ALPHA*(E d)d  == per-row e' = e - (e.d)d)
"kinds=all" = mlp+o+gdn+embed. Module presence is detected at load time, so
"all" covers each layer's actual writer set. The surgery is:
    W_new = W - ALPHA * d (d^T W)   # makes the writer output orthogonal to d
Computed in FP32, stored back in BF16. Only the designed tensors change.

Gate: every non-designed tensor bit-identical vs /out/text-lm (asserted;
exactly the designed set differs).
Output: /out/text-lm-abliterated/ (same shard layout + config + tokenizer).
"""
import gc
import os

import stageutil as su
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM


def parse_kinds(spec):
    """'mlp+o+gdn' / 'mlp, embed' / 'all' -> frozenset; mirrors surgery_sweep."""
    spec = (spec or "").strip() or "mlp"
    kinds = set()
    for tok in __import__("re").split(r"[+,\s|]+", spec):
        if not tok:
            continue
        if tok == "all":
            kinds |= {"mlp", "o", "gdn", "embed"}
        elif tok in ("mlp", "o", "gdn", "embed"):
            kinds.add(tok)
        else:
            su.die(f"SURGERY_KINDS: unknown kind {tok!r} (valid: mlp,o,gdn,embed,all)")
    return frozenset(kinds)


def main():
    with su.StageTimer("apply"):
        # DIRECTION is the .pt to apply (required); SURGERY_LAYERS selects the
        # layers (required with a DIRECTION).
        _dir_env = os.environ.get("DIRECTION", "").strip()
        if not _dir_env:
            su.die("DIRECTION is required (path to the direction .pt)")
        d = torch.load(_dir_env, map_location="cpu")
        L = None
        print(f"direction: {_dir_env}", flush=True)
        model_dir = su.out_path("text-lm")
        out_dir = su.out_path("text-lm-abliterated")
        os.makedirs(out_dir, exist_ok=True)

        # load in the checkpoint's own dtype (bf16 for the real model) so the
        # round-trip is bit-exact for every non-surgery tensor; detect it from
        # the actual shard bytes (config torch_dtype may be absent)
        idx = su.read_json(os.path.join(model_dir, "model.safetensors.index.json"))
        shard_of = idx["weight_map"]
        shards = sorted(set(shard_of.values()))
        with safe_open(os.path.join(model_dir, shards[0]), framework="pt") as _f:
            dtype = _f.get_tensor(next(iter(shard_of))).dtype
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=dtype, trust_remote_code=True
        )  # CPU: no GPU needed for weight surgery
        # SURGERY_LAYERS selects the layers, SURGERY_KINDS the writer modules
        # per layer: "48" | "48,49,50" | "27-60" (inclusive) | "all"
        # (= 1..last; layer 0 stays untouched).
        _alpha_env = os.environ.get("ALPHA", "").strip()
        ALPHA = float(_alpha_env) if _alpha_env else 1.0
        _spec = os.environ.get("SURGERY_LAYERS", "").strip()
        if _spec == "none":
            # explicit no-op: 0 tensors changed (useful to run the merge/gate
            # path against a pristine checkpoint); the full gate still asserts
            # that exactly the (empty) designed set differs.
            surgery_layers = []
        elif _spec:
            n_layers = len(model.model.layers)
            surgery_layers = []
            for part in _spec.split(","):
                part = part.strip()
                if not part:
                    continue
                if part == "all":
                    surgery_layers += list(range(1, n_layers))
                elif "-" in part:
                    a, b = part.split("-", 1)
                    surgery_layers += list(range(int(a), int(b) + 1))
                else:
                    surgery_layers.append(int(part))
            if not surgery_layers:
                su.die(f"SURGERY_LAYERS={_spec!r} resolved to no layers")
            _bad = [l for l in surgery_layers if not 0 <= l < n_layers]
            if _bad:
                su.die(f"SURGERY_LAYERS={_spec!r}: out of range [0,{n_layers - 1}]: {_bad}")
        else:
            if L is None:
                su.die("SURGERY_LAYERS is required when DIRECTION override is set")
            surgery_layers = [L]
        kinds = parse_kinds(os.environ.get("SURGERY_KINDS", "mlp"))
        target_keys = []
        for l in surgery_layers:
            _layer = model.model.layers[l]
            _n0 = len(target_keys)
            if "mlp" in kinds:
                target_keys.append(f"model.layers.{l}.mlp.down_proj.weight")
            if "o" in kinds and hasattr(_layer, "self_attn"):
                target_keys.append(f"model.layers.{l}.self_attn.o_proj.weight")
            if "gdn" in kinds and hasattr(_layer, "linear_attn"):
                target_keys.append(f"model.layers.{l}.linear_attn.out_proj.weight")
            if len(target_keys) == _n0:
                su.die(f"SURGERY_KINDS={'+'.join(sorted(kinds))}: layer {l} has none "
                       f"of the requested writer modules")
        if "embed" in kinds:
            target_keys.append("model.embed_tokens.weight")
        target_keys = list(dict.fromkeys(target_keys))
        for _tk in target_keys:
            if _tk not in shard_of:
                su.die(f"target key {_tk} not in checkpoint weight map")

        # Re-normalize in FP32: best.pt is stored in BF16, whose rounding can
        # drift a unit vector's norm by up to ~0.4%. The surgery projection
        # (W - d(d^T W)) is exact only for a true unit vector, so normalize
        # precisely rather than trusting the stored bf16 norm.
        d32 = d.float()
        # Pre-normalization sanity: a corrupted direction (all-zero -> norm 0,
        # garbage -> norm far from 1) would poison the surgery. BF16 storage
        # drifts a unit vector's norm by up to ~0.4%, so accept [0.9, 1.1].
        # (A NaN norm also fails the range check, so no separate isnan needed.)
        # We do NOT re-check the norm after dividing: FP32 rounding over N~5120
        # is ~sqrt(N)*ULP ~ 1e-5, so a post-division tolerance below that is a
        # knife-edge false positive (see the layer-48 run that died at 1.4e-6).
        pre_norm = d32.norm().item()
        if not (0.9 < pre_norm < 1.1):
            su.die(f"direction not normalizable (stored norm={pre_norm:.6f})")
        d32 = d32 / d32.norm()
        if target_keys:
            d32 = d32.to(model.state_dict()[target_keys[0]].device)

        # Apply the projector to every designed target key:
        #   W <- W - ALPHA * d(d^T W)   (FP32 math, BF16 store)
        # (embed_tokens uses the transposed form E - ALPHA*(E d)d)
        for tk in target_keys:
            W = model.state_dict()[tk]
            W32 = W.float()
            if tk == "model.embed_tokens.weight":
                # E <- E - ALPHA*(E d)d: per-row e' = e - ALPHA*(e.d)*d
                W_new = (W32 - ALPHA * torch.outer(W32 @ d32, d32)).to(W.dtype)
            else:
                W_new = (W32 - ALPHA * torch.outer(d32, d32 @ W32)).to(W.dtype)
            W.copy_(W_new)
            print(
                f"applied: {tk} shape {tuple(W.shape)}, "
                f"max |delta| = {(W_new.float() - W32).abs().max().item():.6g}",
                flush=True,
            )
            del W32, W_new
        print(
            f"surgery summary: {len(target_keys)} tensor(s) on layers "
            f"{surgery_layers}; writers={'+'.join(sorted(kinds))}; ALPHA={ALPHA}",
            flush=True,
        )

        # save with identical shard layout. Keys in the index that the model does
        # NOT instantiate (the 15 mtp.* keys, ignored on load via
        # _keys_to_ignore_on_load_unexpected) are absent from state_dict, so copy
        # them through unchanged from the original shards.
        sd = model.state_dict()
        passthrough = {k: s for k, s in shard_of.items() if k not in sd}
        pt_tensors = {}
        if passthrough:
            by_shard = {}
            for k, s in passthrough.items():
                by_shard.setdefault(s, []).append(k)
            for s, keys in by_shard.items():
                with safe_open(os.path.join(model_dir, s), framework="pt") as f:
                    for k in keys:
                        pt_tensors[k] = f.get_tensor(k)
            print(f"passing through {len(pt_tensors)} non-instantiated keys (mtp.*)", flush=True)
        state_by_shard = {s: {} for s in shards}
        for k, s in shard_of.items():
            state_by_shard[s][k] = sd[k] if k in sd else pt_tensors[k]
        for s, state in state_by_shard.items():
            save_file(state, os.path.join(out_dir, s))
        su.write_json(
            os.path.join(out_dir, "model.safetensors.index.json"),
            {"metadata": idx["metadata"], "weight_map": shard_of},
        )
        for fn in sorted(os.listdir(model_dir)):
            if fn.endswith((".json", ".jinja")) and fn != "model.safetensors.index.json":
                import shutil

                shutil.copy2(os.path.join(model_dir, fn), os.path.join(out_dir, fn))
        del model, sd, state_by_shard
        gc.collect()

        # gate: exactly the DESIGNED tensors differ, all others bit-identical
        diff = []
        for shard in shards:
            with safe_open(os.path.join(model_dir, shard), framework="pt") as a, safe_open(
                os.path.join(out_dir, shard), framework="pt"
            ) as b:
                for k in a.keys():
                    if not torch.equal(a.get_tensor(k), b.get_tensor(k)):
                        diff.append((shard, k))
        expected = {(shard_of[tk], tk) for tk in target_keys}
        if len(diff) != len(expected) or set(diff) != expected:
            su.die(
                f"expected changed set {sorted(expected)}, got: {diff} "
                f"(missing={sorted(expected - set(diff))}, "
                f"unexpected={sorted(set(diff) - expected)})"
            )
        print(
            f"gate passed: exactly {len(diff)} tensor(s) changed "
            f"({sorted(tk for _, tk in diff)}); rest bit-identical"
        )

        with open(os.path.join(out_dir, "COMPLETE"), "w") as f:
            f.write("ok\n")


if __name__ == "__main__":
    main()
