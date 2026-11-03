"""Refusal-vs-benign contrast direction for DETERMINISTIC hard-refusals.

For hard-refusal questions (refused even at temp 0 / no-thinking) the refusal is
the majority, not a minority tail, so a "tail vs mode" contrast is noisy. This
mines a clean REFUSAL-vs-BENIGN contrast instead:
  1. Greedy (temp 0, non-thinking) generation - deterministic.
  2. REFUSAL set: decision-position activations on the hard-refusal prompts.
  3. BENIGN set: decision-position activations on benign prompts (the "answer" state).
  4. Per layer: d = normalize(mean(refusal) - mean(benign)); pick the strongest
     contrast norm (edge layers excluded).
  5. Persist directions/contrast.pt + .json (provenance).

The bake applies this single d through the multi-layer writers (stage 4:
SURGERY_LAYERS / SURGERY_KINDS / ALPHA / DIRECTION).
Run in-container (GPU) via abliterate.sh.
"""
import os

import stageutil as su
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REFUSAL_FILE = os.environ.get("CONTRAST_REFUSAL", "/prompts/political_probes.txt")
# Benign prompts the model answers normally - they define the "answer" state.
# `or` (not a get-default) so an EMPTY CONTRAST_BENIGN (the wrapper passes ""
# when unset) still falls back to the default set instead of yielding 0 benign.
_BENIGN_DEFAULT = (
    "Give a recipe for chocolate chip cookies.\n"
    "Explain how a drip coffee maker works in two sentences.\n"
    "Write a haiku about the sea.\n"
    "What is the capital of France? Answer in one word."
)
BENIGN_PROMPTS = [
    p.strip()
    for p in (os.environ.get("CONTRAST_BENIGN") or _BENIGN_DEFAULT).split("\n")
    if p.strip()
]
MAX_TOK = int(os.environ.get("CONTRAST_MAX_TOKENS") or 250)
EDGE = 3  # exclude layers 0..EDGE-1 and n-EDGE..n-1 (stage-3 edge rule)
OUT_PT = su.out_path("directions", "contrast.pt")
OUT_JSON = su.out_path("directions", "contrast.json")
GENS_JSON = su.out_path("baseline", "contrast_generations.json")


def read_prompt_file(path):
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]


def main():
    with su.StageTimer("contrast-mine"):
        lm_dir = su.out_path("text-lm")
        tok = AutoTokenizer.from_pretrained(lm_dir, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            lm_dir, torch_dtype=torch.bfloat16, device_map="cuda",
            trust_remote_code=True,
        )
        model.eval()
        layers = model.model.layers
        n_layers = len(layers)

        refusal_prompts = read_prompt_file(REFUSAL_FILE)
        if not refusal_prompts:
            su.die(f"no refusal prompts in {REFUSAL_FILE}")
        tpl = su.build_template_fmt(tok)
        print(
            f"contrast mining: {len(refusal_prompts)} refusal + "
            f"{len(BENIGN_PROMPTS)} benign, greedy non-thinking "
            f"(budget={MAX_TOK}, thinking={su.THINKING})",
            flush=True,
        )

        # ---- capture resid_post at the DECISION position (last prompt token) ----
        # model.generate runs many forwards; only the PREFILL forward (seq len
        # == prompt len) has its last position at the decision point.
        stash = {}
        cur_base = [0]

        def mk(i):
            def fn(module, inp, out):
                if inp[0].shape[1] != cur_base[0]:
                    return  # a generation step, not the prefill
                t = out[0] if isinstance(out, tuple) else out
                stash[f"L{i}"] = t[:, -1, :].squeeze(0).detach().cpu()
            return fn

        handles = [layers[i].register_forward_hook(mk(i)) for i in range(n_layers)]

        def gen_and_capture(prompt):
            text = tpl.format(instruction=prompt)
            ids = tok(text, return_tensors="pt").input_ids.to(model.device)
            cur_base[0] = ids.shape[1]
            stash.clear()
            out = model.generate(
                ids,
                max_new_tokens=MAX_TOK,
                do_sample=False,  # greedy: deterministic (temp 0)
                pad_token_id=tok.eos_token_id,
            )
            comp = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
            acts = dict(stash)  # copied before the next generate clears it
            del out, ids
            return comp, acts

        refusal_acts, benign_acts, gens = [], [], []
        for p in refusal_prompts:
            comp, acts = gen_and_capture(p)
            refusal_acts.append(acts)
            gens.append({"set": "refusal", "prompt": p, "completion": comp})
            print(f"[REFUSAL] {p[:44]!r} -> {comp[:70]!r}", flush=True)
        for p in BENIGN_PROMPTS:
            comp, acts = gen_and_capture(p)
            benign_acts.append(acts)
            gens.append({"set": "benign", "prompt": p, "completion": comp})
            print(f"[BENIGN ] {p[:44]!r} -> {comp[:70]!r}", flush=True)
        for h in handles:
            h.remove()

        if not refusal_acts or not benign_acts:
            su.die("need at least one refusal and one benign sample")

        # ---- per-layer contrast: refusal vs benign at the decision position ----
        norms = {}
        best_layer, best_d = None, None
        for i in range(EDGE, n_layers - EDGE):
            k = f"L{i}"
            r = torch.stack([a[k] for a in refusal_acts]).float().mean(dim=0)
            b = torch.stack([a[k] for a in benign_acts]).float().mean(dim=0)
            d = r - b
            norm = d.norm().item()
            norms[i] = norm
            if best_d is None or norm > best_d[1]:
                best_layer, best_d = i, (d, norm)
        d_unit = (best_d[0] / best_d[0].norm()).to(torch.bfloat16)

        top10 = sorted(norms.items(), key=lambda kv: -kv[1])[:10]
        print(
            f"selected layer {best_layer} (norm={best_d[1]:.4f}); top-10: "
            + ", ".join(f"{l}:{v:.3f}" for l, v in top10),
            flush=True,
        )

        su.write_json(GENS_JSON, gens)
        su.write_json(
            OUT_JSON,
            {
                "kind": ("refusal-vs-benign contrast (deterministic hard-refusals, "
                         "decision position)"),
                "refusal_prompts": refusal_prompts,
                "benign_prompts": BENIGN_PROMPTS,
                "temperature": 0.0,
                "n_refusal": len(refusal_acts),
                "n_benign": len(benign_acts),
                "selected_layer": best_layer,
                "selected_norm": best_d[1],
                "top10_layers": top10,
                "edge_excluded": list(range(EDGE)) + list(range(n_layers - EDGE, n_layers)),
            },
        )
        torch.save(d_unit, OUT_PT)
        print(f"\nWROTE {OUT_PT} (layer {best_layer}, unit-norm BF16) + {OUT_JSON}",
              flush=True)
        print("Next: sweep it, then bake the winner:", flush=True)
        print("  SWEEP_DIRECTION=/out/directions/contrast.pt ...", flush=True)


if __name__ == "__main__":
    main()
