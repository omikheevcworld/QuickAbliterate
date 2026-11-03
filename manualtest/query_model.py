#!/usr/bin/env python3
"""Send a prompt to the bare stock-model server (manualtest/start_model.py).

The server must be running first:
    python3 manualtest/start_model.py        # terminal 1
    python3 manualtest/query_model.py "..."  # terminal 2

Usage:
    python3 manualtest/query_model.py "Does the sky exist?"
    python3 manualtest/query_model.py "Tell me about X" --thinking 0 --temperature 0.2
    python3 manualtest/query_model.py "Compare A and B" --samples 3 --max-tokens 1024
    python3 manualtest/query_model.py --prompt-file my_prompt.txt
    echo "your prompt" | python3 manualtest/query_model.py          # stdin

Settings:
    --thinking 1|0     1 = let it think (default); 0 = no thinking block
    --temperature F    0.0 = greedy/deterministic; higher = more random
    --max-tokens N     cap on generated tokens (0 = no cap, run to EOS; default)
    --top-p F          nucleus sampling (default 1.0 = off)
    --samples N        draw N independent samples (n)
    --url URL          server endpoint (default http://127.0.0.1:8080/v1/chat/completions)
"""
import argparse
import json
import sys
import urllib.request

DEFAULT_URL = "http://127.0.0.1:8080/v1/chat/completions"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompt", nargs="*",
                    help="prompt text (or use --prompt-file / stdin)")
    ap.add_argument("--prompt-file", help="read the prompt from a file")
    ap.add_argument("--url", default=DEFAULT_URL, help="server endpoint")
    ap.add_argument("--thinking", default="1", choices=["0", "1"],
                    help="1 = think (default); 0 = no thinking block")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=0,
                    help="cap on generated tokens (0 = no cap, run to EOS; default)")
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--samples", type=int, default=1,
                    help="number of independent samples (n)")
    args = ap.parse_args()

    # resolve the prompt
    if args.prompt_file:
        with open(args.prompt_file) as f:
            prompt = f.read().strip()
    elif args.prompt:
        prompt = " ".join(args.prompt)
    else:
        prompt = sys.stdin.read().strip()
    if not prompt:
        raise SystemExit("FATAL: no prompt (pass as arg, --prompt-file, or stdin)")

    payload = {
        "model": "stock",
        "messages": [{"role": "user", "content": prompt}],
        "thinking": args.thinking == "1",
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "top_p": args.top_p,
        "n": args.samples,
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        args.url, data=data, method="POST",
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=3600) as resp:
            out = json.loads(resp.read().decode())
    except Exception as e:
        raise SystemExit(
            f"FATAL: server error: {type(e).__name__}: {e}\n"
            f"  Is start_model.py running? (endpoint: {args.url})")

    for i, ch in enumerate(out.get("choices", [])):
        msg = ch.get("message", {})
        if args.samples > 1:
            print(f"\n{'=' * 70}\nSAMPLE {i + 1}/{args.samples}\n{'=' * 70}")
        think = msg.get("thinking")
        if think:
            print(f"--- thinking ---\n{think.strip()}\n")
        print(f"--- answer ---\n{msg.get('content', '').strip()}")
    print()


if __name__ == "__main__":
    main()
