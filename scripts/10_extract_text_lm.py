"""Extract a text-only LM from the full VLM checkpoint.

Reads /models/<MODEL_NAME> (the full VLM), writes /out/text-lm/:
  - remapped safetensors: model.language_model.* -> model.* ; lm_head.* / mtp.* kept
    top-level; model.visual.* dropped (same 18-shard layout)
  - config.json from the original text_config + architectures ["Qwen3_5ForCausalLM"]
  - tokenizer/processor files

Gate: loads via AutoModelForCausalLM with zero unexpected keys, and greedy
generation on 3 fixed prompts matches the full VLM's text-only generation
(loaded sequentially, peak ~60GB).
"""
import gc
import os
import re
import shutil

import stageutil as su
import torch
from safetensors import safe_open
from safetensors.torch import save_file

PARITY_PROMPTS = [
    "Explain how a drip coffee maker works in two sentences.",
    "Write a haiku about the sea.",
    "What is the capital of France? Answer in one word.",
]


def detect_lm_prefix(weight_map):
    """Find the dominant LM key prefix, e.g. 'model.language_model.'."""
    from collections import Counter

    keys = [k for k in weight_map if not su.VISUAL_RE.search(k)]
    cands = Counter()
    for k in keys:
        parts = k.split(".")
        if len(parts) >= 3:
            cands[".".join(parts[:2]) + "."] += 1
    if not cands:
        su.die("cannot detect LM prefix from weight map")
    prefix, count = cands.most_common(1)[0]
    frac = count / len(keys)
    if frac < 0.9:
        su.die(f"LM prefix '{prefix}' covers only {frac:.1%} of non-visual keys")
    return prefix, len(keys)


def remap_key(key, prefix):
    if key.startswith(prefix):
        return key.replace(prefix, "model.", 1)
    return key  # lm_head.*, mtp.* stay top-level


def main():
    with su.StageTimer("extract"):
        model_dir = su.models_path(su.MODEL_NAME)
        out_dir = su.out_path("text-lm")
        os.makedirs(out_dir, exist_ok=True)

        index = su.read_json(os.path.join(model_dir, "model.safetensors.index.json"))
        weight_map = index["weight_map"]
        prefix, n_non_visual = detect_lm_prefix(weight_map)
        print(f"LM prefix: '{prefix}' ({n_non_visual} non-visual keys)")

        expected = {remap_key(k, prefix) for k in weight_map if not su.VISUAL_RE.search(k)}
        for must in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"):
            if must not in expected:
                su.die(f"remapped keys missing expected '{must}'")
        cfg = su.read_json(os.path.join(model_dir, "config.json"))
        n_layers_expected = cfg["text_config"]["num_hidden_layers"]
        layer_ids = {
            int(m.group(1))
            for k in expected
            if (m := re.match(r"^model\.layers\.(\d+)\.", k))
        }
        if len(layer_ids) != n_layers_expected:
            su.die(
                f"expected {n_layers_expected} LM layers (per config), found {len(layer_ids)}"
            )
        if min(layer_ids) != 0 or max(layer_ids) != n_layers_expected - 1:
            su.die(
                f"layer indices not contiguous 0..{n_layers_expected - 1}: "
                f"{sorted(layer_ids)[:5]}..."
            )

        # ---- stream shards -> out shards (identical shard boundaries) ----
        shards = sorted(set(weight_map.values()))
        new_weight_map = {}
        total_bytes = 0
        for shard in shards:
            src = os.path.join(model_dir, shard)
            state = {}
            with safe_open(src, framework="pt") as f:
                for key in f.keys():
                    if su.VISUAL_RE.search(key):
                        continue
                    rk = remap_key(key, prefix)
                    t = f.get_tensor(key)
                    state[rk] = t
                    new_weight_map[rk] = shard
                    total_bytes += t.numel() * t.element_size()
            save_file(state, os.path.join(out_dir, shard))
            del state
            gc.collect()
            print(f"  wrote {shard} ({len(new_weight_map)} keys mapped so far)")

        su.write_json(
            os.path.join(out_dir, "model.safetensors.index.json"),
            {"metadata": {"total_size": float(total_bytes)}, "weight_map": new_weight_map},
        )

        # ---- config from text_config ----
        text_cfg = dict(cfg["text_config"])
        text_cfg["architectures"] = ["Qwen3_5ForCausalLM"]
        su.write_json(os.path.join(out_dir, "config.json"), text_cfg)

        # ---- tokenizer / processor files ----
        for fn in sorted(os.listdir(model_dir)):
            if fn.endswith((".json", ".jinja")) and fn not in (
                "config.json",
                "model.safetensors.index.json",
            ):
                shutil.copy2(os.path.join(model_dir, fn), os.path.join(out_dir, fn))

        # ---- gate 1: loads cleanly ----
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(out_dir, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            out_dir, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
        )
        actual = set(model.state_dict().keys())
        missing = expected - actual
        unexpected = actual - expected
        if unexpected:
            su.die(f"unexpected keys after load: {sorted(unexpected)[:10]}")
        if missing:
            if all(k.startswith("mtp.") for k in missing):
                print(
                    f"NOTE: {len(missing)} mtp.* keys not instantiated by "
                    "Qwen3_5ForCausalLM — kept in dir, unused by the text model"
                )
            else:
                su.die(f"missing keys after load: {sorted(missing)[:10]}")
        print("gate 1 passed: text LM loads with zero unexpected keys")

        # ---- gate 2: parity with the full VLM (sequential loads) ----
        def text_gen(model, tok, prompt):
            # transformers 5.x: apply_chat_template(tokenize=True) returns a
            # BatchEncoding, not a tensor — extract input_ids explicitly.
            ids = su.apply_chat_template(
                tok, prompt, tokenize=True, return_tensors="pt"
            )["input_ids"].to(model.device)
            out = model.generate(
                ids, max_new_tokens=64, do_sample=False, pad_token_id=tok.eos_token_id
            )
            return out[0][ids.shape[1]:].tolist()

        text_outs = {p: text_gen(model, tok, p) for p in PARITY_PROMPTS}
        del model, tok
        gc.collect()
        torch.cuda.empty_cache()

        vlm = su.load_vlm(model_dir, torch.bfloat16, "cuda")
        proc = su.load_processor(model_dir)

        def vlm_gen(vlm, proc, prompt):
            text = su.apply_chat_template(proc, prompt, tokenize=False)
            inputs = proc(text=[text], return_tensors="pt").to(vlm.device)
            out = vlm.generate(**inputs, max_new_tokens=64, do_sample=False)
            return out[0][inputs["input_ids"].shape[1]:].tolist()

        vlm_outs = {p: vlm_gen(vlm, proc, p) for p in PARITY_PROMPTS}
        del vlm, proc
        gc.collect()

        parity = {
            p: {"text_lm": text_outs[p], "vlm": vlm_outs[p], "match": text_outs[p] == vlm_outs[p]}
            for p in PARITY_PROMPTS
        }
        su.write_json(su.out_path("extract_parity.json"), parity)
        for p, r in parity.items():
            print(f"  parity[{r['match']}] {p[:40]!r}: {len(r['text_lm'])} vs {len(r['vlm'])} toks")
        if not all(r["match"] for r in parity.values()):
            su.die("parity FAILED — extracted text LM diverges from the VLM text path; see out/extract_parity.json")

        print("gate 2 passed: text-only generation matches the full VLM")
        with open(os.path.join(out_dir, "COMPLETE"), "w") as f:
            f.write("ok\n")


if __name__ == "__main__":
    main()
