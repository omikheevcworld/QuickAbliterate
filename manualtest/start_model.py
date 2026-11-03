#!/usr/bin/env python3
"""Start the STOCK Qwen3.8-27B (text-only) as a bare local model server.

No serving stack, no docker: loads the checkpoint directly with transformers
(the same loading pattern the test harness uses) and serves an OpenAI-compatible
HTTP API on a local port. The model is the stock BF16 text LM at
data/out/text-lm (extracted from HF Qwen/Qwen3.8-27B).

Usage:
    python3 manualtest/start_model.py                          # data/out/text-lm, :8080
    python3 manualtest/start_model.py --port 8090
    python3 manualtest/start_model.py --model-dir ~/models/qwen3.8-27b

Then query it (separate terminal):
    python3 manualtest/query_model.py "your prompt"
    python3 manualtest/query_model.py "your prompt" --thinking 0 --temperature 0.2
"""
import argparse
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# With enable_thinking=True the chat template pre-opens the thinking block in
# the prompt, so the completion starts with thinking content and carries only
# the close tag. Split on the first close tag (matches scripts/stageutil.py).
THINK_CLOSE = "\n</think>"

DEFAULT_MODEL_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "out", "text-lm")
)


class ModelServer:
    def __init__(self, model_dir, max_model_len):
        model_dir = os.path.abspath(model_dir)
        if not os.path.isdir(model_dir):
            raise SystemExit(f"FATAL: model dir not found: {model_dir}")
        print(f"loading {model_dir} (bf16, cuda) ...", flush=True)
        t0 = time.time()
        # No device_map (avoids the `accelerate` dependency); load then move to
        # the GPU. On a unified-memory system this is the same pool either way.
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self.model = self.model.to("cuda")
        self.model.eval()
        self.tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        self.model_dir = model_dir
        self.model_id = os.path.basename(model_dir.rstrip("/"))
        self.max_model_len = max_model_len
        self.lock = threading.Lock()  # single GPU: serialize generation
        print(f"loaded in {time.time() - t0:.1f}s; "
              f"{len(self.model.model.layers)} layers", flush=True)

    def apply_template(self, messages, thinking):
        try:
            return self.tok.apply_chat_template(
                messages, add_generation_prompt=True,
                enable_thinking=thinking, tokenize=False,
            )
        except TypeError:
            # template without an enable_thinking slot
            return self.tok.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )

    def generate(self, messages, thinking, temperature, max_tokens, top_p, n):
        text = self.apply_template(messages, thinking)
        enc = self.tok(text, return_tensors="pt").to(self.model.device)
        input_len = enc["input_ids"].shape[1]
        if input_len >= self.max_model_len:
            raise ValueError(f"prompt too long ({input_len} >= {self.max_model_len})")
        # 0 = no cap: generate until EOS or the context window is full
        room = self.max_model_len - input_len
        max_new = room if max_tokens <= 0 else min(max_tokens, room)
        do_sample = temperature > 0
        out = []
        with self.lock:
            for _ in range(max(1, n)):
                gen = self.model.generate(
                    **enc,
                    max_new_tokens=max_new,
                    do_sample=do_sample,
                    temperature=temperature if do_sample else None,
                    top_p=top_p if (do_sample and 0 < top_p < 1.0) else None,
                )
                comp = self.tok.decode(gen[0][input_len:], skip_special_tokens=True)
                out.append(comp)
                del gen
        results = []
        for comp in out:
            if THINK_CLOSE in comp:
                thinking_part, answer = comp.split(THINK_CLOSE, 1)
            else:
                thinking_part, answer = None, comp
            results.append({"thinking": thinking_part, "answer": answer, "raw": comp})
        return results


SERVER = None  # set in main()


class Handler(BaseHTTPRequestHandler):
    server_version = "BareQwen/1.0"

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):  # keep the console clean
        pass

    def do_GET(self):
        if self.path in ("/health", "/"):
            self._send(200, {"status": "ok", "model": SERVER.model_id})
        elif self.path == "/v1/models":
            self._send(200, {"data": [{"id": SERVER.model_id}]})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self._send(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(length)) if length else {}
        except Exception as e:
            self._send(400, {"error": f"bad request: {e}"})
            return
        messages = req.get("messages")
        if not messages:
            self._send(400, {"error": "missing 'messages'"})
            return
        thinking = bool(req.get("thinking", True))
        temperature = float(req.get("temperature", 0.7))
        max_tokens = int(req.get("max_tokens")
                         or req.get("max_completion_tokens") or 0)
        top_p = float(req.get("top_p", 1.0))
        n = int(req.get("n", 1))
        try:
            results = SERVER.generate(messages, thinking, temperature,
                                      max_tokens, top_p, n)
        except Exception as e:
            self._send(500, {"error": f"{type(e).__name__}: {e}"})
            return
        choices = [
            {"index": i,
             "message": {"role": "assistant", "content": r["answer"],
                         "thinking": r["thinking"]}}
            for i, r in enumerate(results)
        ]
        self._send(200, {"model": SERVER.model_id, "choices": choices})


def main():
    global SERVER
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", default=DEFAULT_MODEL_DIR,
                    help="checkpoint dir (default: data/out/text-lm)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--max-model-len", type=int, default=8192,
                    help="max prompt+generation tokens")
    args = ap.parse_args()

    SERVER = ModelServer(args.model_dir, args.max_model_len)
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"\nREADY: stock {SERVER.model_id} on http://{args.host}:{args.port}",
          flush=True)
    print(f"  health:  curl http://{args.host}:{args.port}/health", flush=True)
    print(f"  query:   python3 manualtest/query_model.py 'your prompt'", flush=True)
    print("  (Ctrl-C to stop)", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping ...", flush=True)
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
