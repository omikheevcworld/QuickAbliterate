"""Merge abliterated LM weights back into the full VLM checkpoint.

Streams the original /models/<MODEL_NAME> shard-by-shard (one tensor in memory
at a time): model.language_model.* keys are replaced with the abliterated
text-LM weights (prefix re-added), everything else (vision tower, lm_head,
mtp.*) is passthrough. Shard names/boundaries are preserved, so the original
index.json stays valid.

Gate: /out/<MODEL_NAME>-abliterated loads via the VLM auto class with full key
alignment and generates for both a text prompt and an image prompt.
"""
import gc
import os
import shutil

import stageutil as su
import torch
from safetensors import safe_open
from safetensors.torch import save_file


def main():
    with su.StageTimer("merge"):
        orig = su.models_path(su.MODEL_NAME)
        abl = su.out_path("text-lm-abliterated")
        out = su.out_path(f"{su.MODEL_NAME}-abliterated")
        os.makedirs(out, exist_ok=True)

        orig_idx = su.read_json(os.path.join(orig, "model.safetensors.index.json"))
        abl_idx = su.read_json(os.path.join(abl, "model.safetensors.index.json"))["weight_map"]

        abl_handles = {}

        def abl_tensor(rkey):
            shard = abl_idx[rkey]
            h = abl_handles.get(shard)
            if h is None:
                h = safe_open(os.path.join(abl, shard), framework="pt")
                abl_handles[shard] = h
            return h.get_tensor(rkey)

        weight_map = orig_idx["weight_map"]
        shards = sorted(set(weight_map.values()))
        n_replaced = 0
        for shard in shards:
            state = {}
            with safe_open(os.path.join(orig, shard), framework="pt") as f:
                for key in f.keys():
                    if key.startswith("model.language_model."):
                        state[key] = abl_tensor(
                            key.replace("model.language_model.", "model.", 1)
                        )
                        n_replaced += 1
                    else:
                        state[key] = f.get_tensor(key)
            save_file(state, os.path.join(out, shard))
            del state
            gc.collect()
            print(f"merged {shard} ({n_replaced} LM tensors replaced so far)", flush=True)

        su.write_json(
            os.path.join(out, "model.safetensors.index.json"),
            {"metadata": orig_idx["metadata"], "weight_map": weight_map},
        )
        for fn in sorted(os.listdir(orig)):
            if fn.endswith((".json", ".jinja")) and fn != "model.safetensors.index.json":
                shutil.copy2(os.path.join(orig, fn), os.path.join(out, fn))
        print(f"merge done: {n_replaced} LM tensors replaced, vision tower passthrough")

        # ---- gate: load + generate (text + image) ----
        from PIL import Image
        from transformers import AutoProcessor

        model = su.load_vlm(out, torch.bfloat16, "cuda")
        proc = AutoProcessor.from_pretrained(out, trust_remote_code=True)

        actual = set(model.state_dict().keys())
        expected = set(weight_map.keys())
        # keys the VLM ignores on load (mtp.*) are in the index + shards but not
        # instantiated — expected to be absent from state_dict. Use the model's
        # own ignore patterns so this stays correct if they change.
        import re as _re

        ignore = getattr(model, "_keys_to_ignore_on_load_unexpected", [])
        pats = [_re.compile(p) for p in ignore]
        expected = {k for k in expected if not any(p.search(k) for p in pats)}
        if actual != expected:
            su.die(
                f"key mismatch after load: missing={sorted(expected - actual)[:5]} "
                f"unexpected={sorted(actual - expected)[:5]}"
            )

        text = su.apply_chat_template(
            proc, "Reply with the single word: hello", tokenize=False
        )
        inputs = proc(text=[text], return_tensors="pt").to(model.device)
        gen = model.generate(**inputs, max_new_tokens=32, do_sample=False)
        dec = proc.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"text generation: {dec[:120]!r}")

        img = Image.new("RGB", (256, 256), (120, 160, 220))
        im_msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Describe this image in one sentence."},
                ],
            }
        ]
        try:
            im_text = proc.apply_chat_template(
                im_msgs, add_generation_prompt=True, enable_thinking=False, tokenize=False
            )
        except TypeError:
            im_text = proc.apply_chat_template(
                im_msgs, add_generation_prompt=True, tokenize=False
            )
        im_inputs = proc(text=[im_text], images=[img], return_tensors="pt").to(model.device)
        gen2 = model.generate(**im_inputs, max_new_tokens=64, do_sample=False)
        dec2 = proc.decode(gen2[0][im_inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"image generation: {dec2[:120]!r}")

        with open(os.path.join(out, "COMPLETE"), "w") as f:
            f.write("ok\n")


if __name__ == "__main__":
    main()
