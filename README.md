# quick-abliteration

A minimal, self-contained pipeline that **abliterates** a model — makes it stop
refusing a set of hard-refusal questions — via a **refusal-vs-benign contrast**
direction. One command runs the whole thing:

```bash
./abliterate.sh
```

It mines the refusal direction, sweeps surgery designs at runtime, bakes the
winner into the full checkpoint, and verifies the result. No judge, no
multi-step orchestration — just the technique.

## How it works

Censorship in these models is a **direction** in the residual stream. For a
deterministic hard-refusal (refused even at temperature 0 / no-thinking) the
refusal is the *dominant* mode, so the direction is mined as a clean
**refusal-vs-benign contrast**: decision-position residual activations on the
refusal prompts minus the same on benign prompts the model answers normally.
The single unit direction is then projected out of the residual stream at the
selected writer modules (`mlp.down_proj` + `self_attn.o_proj` +
`linear_attn.out_proj`) across the designed layer range, and merged back into
the full checkpoint (vision tower untouched).

**Regime:** zero-thinking + temperature 0 (greedy) throughout, so a refusal is
deterministic and reproducible. Refusal detection is a hard-stop regex
("I can't / I cannot / I'm not able / as an AI …", EN + ZH) — no LLM judge.

**Scope:** stages 0 (extract) and 4 (merge) assume a vision-language model — a
text LM nested under `model.language_model.*` with a `model.visual.*` tower (as
in Qwen3.8-27B). For a pure text model, skip those two stages and run
mine → sweep → apply directly on the model.

## Requirements

- **Docker** (with the NVIDIA runtime for `--gpus`).
- **~120 GB unified RAM** (GPU + system memory shared) to load the base model
  and run the surgery at peak.
- **Disk:** the base model (~55 GB) + the abliterated output (~55 GB) +
  intermediate text-LM copies.
- **Host Python 3** with `huggingface_hub` (for the optional model download).
- **No serving stack (SGLang / vLLM / …) required** — the pipeline runs its own
  short-lived containers.

> **Platform note:** the `Dockerfile` uses an NGC PyTorch image
> (`nvcr.io/nvidia/pytorch:26.09-py3`) built for the GB10 (aarch64, CUDA 13,
> sm_121). On a different platform (e.g. x86_64) change the `FROM` line to a
> matching NGC PyTorch tag and rebuild.

## Usage

```bash
# defaults: Qwen/Qwen3.8-27B from ~/models/qwen3.8-27b, prompts/political_probes.txt
./abliterate.sh

# host-side pre-checks only (no GPU stages) - run this first if you like
./abliterate.sh --preflight

# reuse an existing data/out/text-lm (skip the ~6-min stage-0 extraction)
./abliterate.sh --skip-extract

# override any of them:
MODEL_REPO=Qwen/Qwen3.8-27B \
MODEL_DIR=$HOME/models/qwen3.8-27b \
PROMPTS_FILE=$PWD/prompts/political_probes.txt \
SWEEP_CONFIGS="1-63@1.0/mlp+o+gdn,1-63@1.2/mlp+o+gdn" \
./abliterate.sh

# re-run every stage from scratch (ignore the done-markers)
FORCE=1 ./abliterate.sh
```

If the model is **not** already at `MODEL_DIR`, it is downloaded from
`MODEL_REPO` (Hugging Face) first; if it **is** there, the download is skipped.

### Env knobs

| var | default | meaning |
|---|---|---|
| `MODEL_REPO` | `Qwen/Qwen3.8-27B` | HF repo to download (if the model is missing) |
| `MODEL_DIR` | `~/models/qwen3.8-27b` | where the base model lives / is downloaded |
| `PROMPTS_FILE` | `prompts/political_probes.txt` | target refusal set (one per line) |
| `SANITY` | cookies recipe | harmless sanity question (auto-appended) |
| `SWEEP_CONFIGS` | `1-63@1.0/…, 1.1/…, 1.2/mlp+o+gdn` | surgery designs to sweep |
| `MAX_WINNER_REFUSED` | `0` | max refusals the winner may leave (0 = all must flip) |
| `FORCE` | `0` | `1` = ignore done-markers and re-run every stage |

## Stages

| stage | what | gate |
|---|---|---|
| 0 extract | text-LM extraction from the VLM (idempotent) | loads + parity |
| 1 baseline | probe the stock base (greedy) | refusal signal present |
| 2 mine | refusal-vs-benign contrast direction | direction valid |
| 3 sweep | runtime sweep over layer × α × writer-kind configs | a sane config flips the targets |
| 4 bake | apply + merge the winner | designed-set bit-diff + load |
| 5 acceptance | probe the abliterated model | big drop from baseline |

The run **stops at the first failed gate** — there is no bake without a passing
sweep.

### Preflight & resume

- **`./abliterate.sh --preflight`** runs host-side checks only — python syntax,
  docker, image build + deps, model presence (or a downloader), a non-empty
  prompts file, and a gate dry-run. It never touches the GPU, so it rules out
  the common "breaks in the middle" causes without spending GPU time. (It cannot
  guarantee the GPU stages succeed — only running them can do that.)
- **Resume:** each stage writes a done-marker to `data/out/.state/` once its work
  **and** gate succeed. Re-running `./abliterate.sh` skips the finished stages and
  resumes from the first unfinished one. To redo a single stage,
  `rm data/out/.state/<stage>`; to redo everything, `FORCE=1 ./abliterate.sh`.
- **`./abliterate.sh --skip-extract`** reuses an existing `data/out/text-lm/` and
  skips stage 0. The text-LM is prompt-independent (only the base model feeds it),
  so it's safe to reuse across prompt changes — saves the ~6-min extraction. Fails
  loudly if `data/out/text-lm/COMPLETE` is missing.

## Output

- `data/out/<model>-abliterated/` — the abliterated full checkpoint.
- `data/out/directions/contrast.pt` (+ `.json` provenance) — the mined direction.
- `data/out/{baseline,sweep,probe}/…json` — the per-stage probe/sweep dumps.

## Try it interactively

Serve the abliterated checkpoint with a bare transformers server (no Docker):

```bash
python3 manualtest/start_model.py --model-dir data/out/<model>-abliterated
python3 manualtest/query_model.py "your question" --thinking 0 --temperature 0
```

## Layout

```
quick-abliteration/
├── abliterate.sh              # the one-liner
├── Dockerfile                 # NGC pytorch image + pinned requirements
├── prompts/political_probes.txt  # target refusal set (edit this)
├── scripts/
│   ├── requirements.txt
│   ├── stageutil.py           # helpers: paths, refusal regex, template, timer
│   ├── 10_extract_text_lm.py  # VLM -> text-LM extraction
│   ├── 73_mine_contrast.py    # refusal-vs-benign contrast miner
│   ├── 40_apply.py            # directional surgery (designed set)
│   └── 50_merge.py            # merge into the full VLM checkpoint
├── tests/
│   ├── probe_checked.py       # refusal prober (baseline + acceptance)
│   ├── surgery_sweep.py       # runtime sweep
│   └── gate.py                # gates (baseline / contrast / sweep / accept)
└── manualtest/
    ├── start_model.py         # bare server (stock or abliterated)
    └── query_model.py         # query client
```
