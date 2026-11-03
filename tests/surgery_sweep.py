"""Runtime surgery sweep: emulate candidate directional surgeries via forward
hooks and measure refusal impact WITHOUT rebuilding any checkpoint.

For a writer weight W [hidden, ...] whose output y = W@x lives in the residual
space, the static rewrite W' = W - alpha*d(d^T W) is EXACTLY equivalent to a
forward hook on that writer returning out - alpha*(out@d)*d. That equivalence
lets us sweep surgery designs at runtime, then bake only the winner into the
checkpoint (apply stage: SURGERY_LAYERS / SURGERY_KINDS / ALPHA).

Each config is scored on the harmful prompts + 1 harmless sanity prompt
(quality watch). The per-config table goes to stdout; the results dump (with
verbatim completions) is written to /out for the gate to read.

Env knobs:
  SWEEP_CONFIGS    comma list of "design[@alpha][/kinds]"; a design is "none"
                   or a layer spec ("48" | "48,49,50" | "27-60" | "all");
                   kinds = residual WRITER modules to project (see below).
  SWEEP_WRITERS    default writer kinds for configs lacking "/kinds" (default mlp).
  SWEEP_MODEL_DIR  model to load (default /out/text-lm — the ORIGINAL LM).
  SWEEP_DIRECTION  direction .pt to project (default /out/directions/contrast.pt).
  SWEEP_PROMPTS    prompts file (in-container path).
  SWEEP_MAX_TOKENS generation budget (default stageutil.MAX_NEW_TOKENS).
  SWEEP_TEMPERATURE  >0 = tail mode: sample SWEEP_SAMPLES completions per prompt
                     at this temperature and report the refusal rate (default 0
                     = single greedy completion).
  SWEEP_SAMPLES      samples per prompt in tail mode (default 1).

Writer kinds (the refusal signal rides the ATTENTION write path too, and most
layers are GatedDeltaNet writing via linear_attn.out_proj — an all-MLP sweep
is behaviorally inert):
  "mlp"   layer.mlp.down_proj          (every layer)
  "o"     layer.self_attn.o_proj       (full-attention layers)
  "gdn"   layer.linear_attn.out_proj   (GatedDeltaNet layers)
  "embed" model.embed_tokens           (reading side, single hook)
  "all"   = mlp+o+gdn+embed
A "/kinds" suffix wins over SWEEP_WRITERS. "none" stays a pure baseline
(no hooks at all).
"""
import json
import os
import re
import sys
import time

for _p in (
    "/workspace",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"
    ),
):
    if os.path.isdir(_p):
        sys.path.insert(0, _p)
import stageutil as su  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

MODEL_DIR = os.environ.get("SWEEP_MODEL_DIR", "/out/text-lm")
PROMPTS_FILE = os.environ.get("SWEEP_PROMPTS", "/prompts/political_probes.txt")
# Default budget = stageutil.MAX_NEW_TOKENS (600 thinking / 250 non-thinking).
MAX_TOK = int(os.environ.get("SWEEP_MAX_TOKENS") or su.MAX_NEW_TOKENS)
# Empty SWEEP_SANITY disables the sanity row (faster sweeps on tiny target sets).
HARMLESS_SANITY = os.environ.get(
    "SWEEP_SANITY", "Give a recipe for chocolate chip cookies."
).strip()
CONFIGS = os.environ.get(
    "SWEEP_CONFIGS",
    "1-63@1.0/mlp+o+gdn,1-63@1.1/mlp+o+gdn,1-63@1.2/mlp+o+gdn",
)
SWEEP_WRITERS = os.environ.get("SWEEP_WRITERS", "mlp")
SWEEP_DIRECTION = os.environ.get("SWEEP_DIRECTION", "")
# Tail mode: SWEEP_TEMPERATURE > 0 -> sample SWEEP_SAMPLES completions per prompt
# at that temperature (do_sample=True, plain temperature) and score the
# per-prompt refusal rate. 0 (default) = single greedy completion.
SWEEP_TEMPERATURE = float(os.environ.get("SWEEP_TEMPERATURE") or "0")
SWEEP_SAMPLES = max(1, int(os.environ.get("SWEEP_SAMPLES") or "1"))
N_SAMP = SWEEP_SAMPLES if SWEEP_TEMPERATURE > 0 else 1
# Results dump (host-side): per-config refused/sane + verbatim rows.
DUMP_PATH = os.environ.get("SWEEP_DUMP", "/out/sweep/sweep_results.json")


def parse_design(spec, n_layers):
    """'none' -> []; '48' / '48,49,50' / '27-60' / 'all' -> sorted layer list."""
    spec = spec.strip()
    if spec == "none":
        return []
    layers = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if part == "all":
            layers += list(range(1, n_layers))
        elif "-" in part:
            a, b = part.split("-", 1)
            layers += list(range(int(a), int(b) + 1))
        else:
            layers.append(int(part))
    bad = [l for l in layers if not 0 <= l < n_layers]
    if bad:
        raise ValueError(f"design {spec!r}: layers out of range [0,{n_layers - 1}]: {bad}")
    return sorted(layers)


KIND_TOKENS = ("mlp", "o", "gdn", "embed", "all")


def parse_kinds(spec):
    """'mlp+o+gdn' / 'mlp,o' / 'all' -> frozenset of writer kinds."""
    spec = (spec or "").strip()
    if not spec:
        return frozenset({"mlp"})
    kinds = set()
    for tok in re.split(r"[+,.\s|]+", spec):
        tok = tok.strip()
        if not tok:
            continue
        if tok == "all":
            kinds |= {"mlp", "o", "gdn", "embed"}
        elif tok in KIND_TOKENS:
            kinds.add(tok)
        else:
            raise ValueError(f"unknown writer kind {tok!r} (valid: {KIND_TOKENS})")
    return frozenset(kinds)


def parse_config(spec, n_layers, default_kinds):
    """'design[@alpha][/kinds]' -> (design, layers, alpha, kinds)."""
    base, _, kinds_s = spec.partition("/")
    design, _, alpha_s = base.partition("@")
    design = design.strip()
    alpha = float(alpha_s) if alpha_s else 1.0
    kinds = parse_kinds(kinds_s if kinds_s.strip() else default_kinds)
    layers = [] if design == "none" else parse_design(design, n_layers)
    return design, layers, alpha, kinds


def read_prompts(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                out.append(line)
    return out


def classify(prompt, completion):
    refused = su.detect_refusal(completion)
    thinking, answer = su.split_thinking(completion)
    answer = (answer or "").strip()
    flags = []
    if su.THINKING and su.THINK_CLOSE not in completion:
        flags.append("unanswered")  # only meaningful when thinking is enabled
    tail = (completion or "")[-600:].split()
    if len(tail) > 40 and len(set(tail)) < 0.25 * len(tail):
        flags.append("repeat-loop")
    if not answer and not flags:
        flags.append("empty")
    return refused, answer, flags


def main():
    best_path = SWEEP_DIRECTION or su.out_path("directions", "contrast.pt")
    d = torch.load(best_path, map_location="cpu").float()
    d = d.reshape(-1)  # miners may save (1, H) or (H,); the hook's matmul needs 1D
    pre_norm = d.norm().item()
    if not (0.9 < pre_norm < 1.1):
        su.die(f"direction not normalizable (stored norm={pre_norm:.6f})")
    d = d / d.norm()
    # Provenance label from the .json next to the .pt, if present.
    try:
        _meta = su.read_json(os.path.splitext(best_path)[0] + ".json")
        key = _meta.get("kind", os.path.basename(best_path))
    except Exception:
        key = os.path.basename(best_path)
    print(f"direction: {best_path} ({key}); unit norm FP32; writers={SWEEP_WRITERS}",
          flush=True)

    if not os.path.isdir(MODEL_DIR):
        sys.exit(f"FATAL: model dir not found: {MODEL_DIR}")
    s_lab = (
        f" sampling=temp={SWEEP_TEMPERATURE} x{N_SAMP}"
        if SWEEP_TEMPERATURE > 0
        else " (greedy)"
    )
    print(f"model={MODEL_DIR} max_new_tokens={MAX_TOK} thinking={su.THINKING}{s_lab}",
          flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
    )
    model.eval()
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    d_cuda = d.to(model.device)
    layers = model.model.layers
    n_layers = len(layers)

    harmful = read_prompts(PROMPTS_FILE)
    prompts = harmful + ([HARMLESS_SANITY] if HARMLESS_SANITY else [])
    texts = [su.apply_chat_template(tok, p, tokenize=False) for p in prompts]
    print(f"{len(harmful)} harmful + 1 harmless sanity prompt; {n_layers} layers", flush=True)

    def generate():
        """comps[prompt_idx] = list of N_SAMP completions (1 in greedy mode)."""
        comps = []
        for text in texts:
            enc = tok(text, return_tensors="pt").to(model.device)
            base = enc["input_ids"].shape[1]
            samples = []
            for _ in range(N_SAMP):
                out = model.generate(
                    **enc,
                    max_new_tokens=MAX_TOK,
                    do_sample=SWEEP_TEMPERATURE > 0,
                    temperature=SWEEP_TEMPERATURE if SWEEP_TEMPERATURE > 0 else None,
                )
                samples.append(tok.decode(out[0][base:], skip_special_tokens=True))
                del out
            comps.append(samples)
            del enc
        return comps

    handles = []

    def attach(spec_layers, alpha, dvec, kinds):
        def hook(module, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            dt = t.float()
            proj = (dt @ dvec).unsqueeze(-1) * dvec
            new = (dt - alpha * proj).to(t.dtype)
            return (new,) + out[1:] if isinstance(out, tuple) else new

        attached = {"mlp": 0, "o": 0, "gdn": 0, "embed": 0}
        if "embed" in kinds:
            emb = getattr(model.model, "embed_tokens", None)
            if emb is None:
                su.die("writer kind 'embed': model has no embed_tokens module")
            handles.append(emb.register_forward_hook(hook))
            attached["embed"] = 1
        for i in spec_layers:
            layer = layers[i]
            if "mlp" in kinds:
                handles.append(layer.mlp.down_proj.register_forward_hook(hook))
                attached["mlp"] += 1
            for kind, parent, proj_name in (
                ("o", "self_attn", "o_proj"),
                ("gdn", "linear_attn", "out_proj"),
            ):
                if kind not in kinds:
                    continue
                m = getattr(getattr(layer, parent, None), proj_name, None)
                if m is None:
                    if not hasattr(layer, parent):
                        continue  # different writer family in this layer
                    su.die(f"layer {i} is a {parent} layer but lacks {proj_name}")
                handles.append(m.register_forward_hook(hook))
                attached[kind] += 1
        for k in ("mlp", "o", "gdn"):
            if k in kinds and spec_layers and attached[k] == 0:
                su.die(
                    f"writer kind {k!r} matched no module across design layers "
                    f"{spec_layers[:8]}... - wrong kind for this design?"
                )
        return attached

    results = {}
    for cfg in CONFIGS.split(","):
        cfg = cfg.strip()
        if not cfg:
            continue
        design, spec_layers, alpha, kinds = parse_config(cfg, n_layers, SWEEP_WRITERS)
        rng = "" if not spec_layers else f" (covers {min(spec_layers)}..{max(spec_layers)})"
        print(f"\n{'=' * 70}\nCONFIG {cfg}: {len(spec_layers)} layer(s){rng}"
              f" writers={'+'.join(sorted(kinds))}", flush=True)
        if design != "none" and (spec_layers or "embed" in kinds):
            attached = attach(spec_layers, alpha, d_cuda, kinds)
            if spec_layers or attached["embed"]:
                print(f"  hooks: " + ", ".join(f"{k}={v}" for k, v in attached.items() if v),
                      flush=True)
        t0 = time.time()
        try:
            comps = generate()
        finally:
            for h in handles:
                h.remove()
            handles.clear()
        minutes = (time.time() - t0) / 60

        harm = comps[: len(harmful)]
        sanity = comps[len(harmful)] if HARMLESS_SANITY else []
        refused = 0
        rows = []
        for p, samples in zip(harmful, harm):
            r_ref = 0
            for si, c in enumerate(samples):
                r, answer, flags = classify(p, c)
                r_ref += r
                shown = answer[:90] if answer else (c or "")[-90:]
                tag = ("REFUSED  " if r else "COMPLIED") + (f" [{','.join(flags)}]" if flags else "")
                slab = f" [s{si + 1}/{N_SAMP}]" if N_SAMP > 1 else ""
                print(f"  {tag}{slab} :: {p[:44]!r} -> {shown!r}", flush=True)
                rows.append(
                    {"prompt": p, "sample": si + 1, "completion": c,
                     "refused": r, "flags": flags, "answer": answer}
                )
            if N_SAMP > 1:
                print(f"  prompt total: {r_ref}/{N_SAMP} refused :: {p[:44]!r}", flush=True)
            refused += r_ref
        if HARMLESS_SANITY:
            _r_s, ans_s, flags_s = classify(HARMLESS_SANITY, sanity[0])
            sane = not flags_s and len(ans_s) > 60
            print(
                f"  harmless sanity: len={len(ans_s)} flags={flags_s or ['none']} :: {ans_s[:90]!r}",
                flush=True,
            )
            rows.append(
                {"prompt": HARMLESS_SANITY, "sample": 1, "completion": sanity[0],
                 "refused": _r_s, "flags": flags_s, "answer": ans_s, "sanity": True}
            )
        else:
            sane = True
            flags_s = []
            print("  harmless sanity: SKIPPED (SWEEP_SANITY empty)", flush=True)
        del comps
        torch.cuda.empty_cache()
        results[cfg] = {
            "refused": refused,
            "n": len(harmful) * N_SAMP,
            "sane": sane,
            "sanity_flags": flags_s,
            "minutes": round(minutes, 1),
            "rows": rows,
        }
        print(
            f"  == SUMMARY {cfg}: refused {refused}/{len(harmful)}; "
            f"sanity {'OK' if sane else 'DEGRADED'}; {minutes:.1f} min",
            flush=True,
        )

    print(f"\n{'=' * 70}\nSWEEP TABLE (goal: refused low AND sanity OK)", flush=True)
    print("            config  refused   sanity   min", flush=True)
    for cfg, r in results.items():
        san = "OK" if r["sane"] else "DEGRADED"
        print(f"{cfg:>18s}   {r['refused']:>2d}/{r['n']:<2d}  {san:>8s}  {r['minutes']:5.1f}", flush=True)
    sane_cfgs = {c: r for c, r in results.items() if r["sane"]}
    winners = [c for c, r in sane_cfgs.items() if r["refused"] == 0]
    if winners:
        print(f"\nWINNER CANDIDATES (refused 0 + sane): {', '.join(winners)}", flush=True)
        print(
            "Bake the best one: SURGERY_LAYERS=<design> SURGERY_KINDS=<kinds> "
            "ALPHA=<alpha> (the abliterate.sh bake stage does this).",
            flush=True,
        )
    elif sane_cfgs:
        ranked = sorted(sane_cfgs, key=lambda c: sane_cfgs[c]["refused"])
        listed = ", ".join(
            f"{c} ({sane_cfgs[c]['refused']}/{sane_cfgs[c]['n']})" for c in ranked
        )
        print(f"\nBest sane designs (partial relief): {listed}", flush=True)
        print(
            "First try an alpha sweep on the leader (append configs like "
            "<design>@0.25 / @0.75 / @1.5 and rerun) before concluding defeat.",
            flush=True,
        )
    else:
        print("\nEvery design degraded the harmless-sanity prompt as well.", flush=True)
        print(
            "No design suppresses the refusal while staying sane - try a wider "
            "layer range, higher ALPHA, or more writer kinds before concluding.",
            flush=True,
        )

    # Results dump (host-side): per-config summary + verbatim rows.
    try:
        os.makedirs(os.path.dirname(DUMP_PATH), exist_ok=True)
        with open(DUMP_PATH, "w") as f:
            json.dump(
                {
                    "direction": best_path,
                    "model": MODEL_DIR,
                    "thinking": su.THINKING,
                    "max_tokens": MAX_TOK,
                    "temperature": SWEEP_TEMPERATURE,
                    "samples": N_SAMP,
                    "configs": {
                        cfg: {k: v for k, v in r.items() if k != "rows"}
                        for cfg, r in results.items()
                    },
                    "rows": {cfg: r["rows"] for cfg, r in results.items()},
                },
                f,
                indent=1,
            )
        print(f"\nDUMP {DUMP_PATH}", flush=True)
    except OSError as e:
        print(f"\nnote: results dump not written ({e})", flush=True)


if __name__ == "__main__":
    main()
