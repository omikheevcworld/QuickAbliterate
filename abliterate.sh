#!/usr/bin/env bash
# quick-abliteration - one-liner to abliterate a model via refusal-vs-benign
# contrast. Mines the refusal direction, sweeps surgery designs at runtime,
# bakes the winner into the full checkpoint, and verifies the result.
#
#   ./abliterate.sh                run (resumes from the first unfinished stage)
#   ./abliterate.sh --preflight    host-side checks only (no GPU stages)
#   ./abliterate.sh --skip-extract reuse an existing data/out/text-lm (skip stage 0)
#   MODEL_REPO=... MODEL_DIR=... PROMPTS_FILE=... ./abliterate.sh
#   FORCE=1 ./abliterate.sh        ignore done-markers; re-run every stage
#
# Stages (gated; stops at the first failed gate; each is marked done on success
# so a re-run resumes from the first unfinished stage):
#   0 extract    text-LM extraction from the VLM
#   1 baseline   probe the stock base (greedy)
#   2 mine       refusal-vs-benign contrast direction
#   3 sweep      runtime sweep over layer-range x alpha x writer-kind configs
#   4 bake       apply + merge the winner into the full VLM checkpoint
#   5 acceptance probe the abliterated model
set -euo pipefail
cd "$(dirname "$0")"

# ---- config (env-overridable) ----
MODEL_REPO="${MODEL_REPO:-Qwen/Qwen3.8-27B}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/qwen3.8-27b}"
MODEL_ROOT="${MODEL_DIR%/*}"                       # parent dir, mounted at /models
MODEL_NAME="${MODEL_NAME:-$(basename "$MODEL_DIR")}"
PROMPTS_FILE="${PROMPTS_FILE:-$PWD/prompts/political_probes.txt}"
PROMPTS="/prompts/$(basename "$PROMPTS_FILE")"     # in-container path
SANITY="${SANITY:-Give a recipe for chocolate chip cookies.}"
CFGS="${SWEEP_CONFIGS:-1-63@1.0/mlp+o+gdn,1-63@1.1/mlp+o+gdn,1-63@1.2/mlp+o+gdn}"
MAX_WINNER_REFUSED="${MAX_WINNER_REFUSED:-0}"      # 0 = the winner must flip every target
IMAGE="quick-abliterate:26.09"
STATE_DIR="data/out/.state"                        # per-stage done-markers (resume)
FORCE="${FORCE:-0}"                                # FORCE=1 re-runs every stage

# host paths for gates; /out paths for containers
BASE_DUMP="data/out/baseline/probe_results.json"
SWEEP_DUMP="data/out/sweep/sweep_results.json"
POST_DUMP="data/out/probe/probe_results.json"
CONTRAST_JSON="data/out/directions/contrast.json"
ABL_OUT="/out/${MODEL_NAME}-abliterated"

mkdir -p data/out "$STATE_DIR"

# ---- common docker-run preamble ----
run_gpu() { # $1 = in-container script path; remaining args = extra -e KEY=VAL
  local script="$1"; shift
  docker run --rm --entrypoint python3 --gpus all \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e OUT_DIR=/out -e MODELS_DIR=/models -e MODEL_NAME="$MODEL_NAME" -e THINKING=0 \
    -v "$PWD"/scripts:/workspace:ro \
    -v "$PWD"/tests:/tests:ro \
    -v "$PWD"/data/out:/out \
    -v "$PWD"/prompts:/prompts:ro \
    -v "$MODEL_ROOT":/models:ro \
    "$@" \
    "$IMAGE" "$script"
}

# ---- stage done-markers (resume) ----
# A stage is skipped if its marker exists (and FORCE!=1). The marker is written
# only AFTER the stage's work + gate succeed, so a re-run resumes from the first
# unfinished stage. FORCE=1 (or `rm data/out/.state/<stage>`) redoes a stage.
stage_done() { [ "$FORCE" = "1" ] && return 1; [ -f "$STATE_DIR/$1" ]; }
mark_done()  { touch "$STATE_DIR/$1"; }

# ---- preflight: host-side checks that rule out most breakage before GPU time ----
# Catches: python syntax, missing docker, image build, image deps, a missing
# model with no downloader, an empty prompts file, and a broken gate. It does
# NOT run the GPU stages, so it cannot guarantee they succeed - but it rules out
# the common "breaks in the middle" causes.
preflight() {
  echo "====================================================================="
  echo "PREFLIGHT  (host-side checks; no GPU stages run)"
  echo "  model=$MODEL_NAME  prompts=$PROMPTS_FILE  image=$IMAGE"
  echo "====================================================================="
  local fail=0 tmp g

  if python3 -m py_compile scripts/*.py tests/*.py manualtest/*.py 2>/dev/null; then
    echo "  [ok]   python syntax (scripts / tests / manualtest)"
  else
    echo "  [FAIL] python syntax"; fail=1
  fi

  if command -v docker >/dev/null 2>&1; then
    echo "  [ok]   docker present"
  else
    echo "  [FAIL] docker not found"; fail=1
  fi

  if docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "  [ok]   image $IMAGE present"
  elif docker build -t "$IMAGE" .; then
    echo "  [ok]   image $IMAGE built"
  else
    echo "  [FAIL] image build failed"; fail=1
  fi

  if docker run --rm --entrypoint python3 "$IMAGE" \
      -c "import torch, transformers, huggingface_hub; \
print('        torch', torch.__version__, '| transformers', transformers.__version__)" \
      >/dev/null 2>&1; then
    echo "  [ok]   image deps importable (torch / transformers / huggingface_hub)"
  else
    echo "  [FAIL] image deps missing"; fail=1
  fi

  if [ -f "$MODEL_DIR/config.json" ] && ls "$MODEL_DIR"/*.safetensors >/dev/null 2>&1; then
    echo "  [ok]   model present at $MODEL_DIR"
  elif python3 -c "import huggingface_hub" >/dev/null 2>&1; then
    echo "  [ok]   model missing; huggingface_hub available (will download at run time)"
  else
    echo "  [FAIL] model missing and huggingface_hub not installed on host"; fail=1
  fi

  if [ -f "$PROMPTS_FILE" ] && grep -vE '^[[:space:]]*(#|$)' "$PROMPTS_FILE" | grep -q .; then
    echo "  [ok]   prompts file has >=1 prompt: $PROMPTS_FILE"
  else
    echo "  [FAIL] prompts file missing or empty: $PROMPTS_FILE"; fail=1
  fi

  tmp="$(mktemp -d)"
  printf '%s\n' '{"rows": [{"prompt": "t", "temp": 0, "refused": true, "sanity": false}]}' > "$tmp/base.json"
  printf '%s\n' '{"n_refusal": 1, "n_benign": 4, "selected_layer": 30, "selected_norm": 1.5}' > "$tmp/contrast.json"
  printf '%s\n' '{"configs": {"1-63@1.0/mlp+o+gdn": {"refused": 0, "n": 1, "sane": true}}}' > "$tmp/sweep.json"
  printf '%s\n' '{"rows": [{"prompt": "t", "temp": 0, "refused": false, "sanity": false}]}' > "$tmp/post.json"
  for g in \
      "baseline $tmp/base.json 1" \
      "contrast $tmp/contrast.json 1 1" \
      "sweep $tmp/sweep.json 0" \
      "accept $tmp/post.json $tmp/base.json"; do
    if python3 tests/gate.py $g >/dev/null 2>&1; then
      echo "  [ok]   gate: $g"
    else
      echo "  [FAIL] gate: $g"; fail=1
    fi
  done
  rm -rf "$tmp"

  echo "====================================================================="
  if [ "$fail" = "0" ]; then
    echo "PREFLIGHT PASS - host-side checks OK (GPU stages NOT run)"
    echo "  Next: ./abliterate.sh   (FORCE=1 ignores any done-markers)"
    exit 0
  else
    echo "PREFLIGHT FAIL - fix the [FAIL] items above, then re-run."
    exit 1
  fi
}

# ---- command-line flags ----
SKIP_EXTRACT="${SKIP_EXTRACT:-0}"   # --skip-extract / SKIP_EXTRACT=1: reuse existing text-lm
DO_PREFLIGHT=0
for arg in "$@"; do
  case "$arg" in
    --preflight|check) DO_PREFLIGHT=1 ;;
    --skip-extract)    SKIP_EXTRACT=1 ;;
    --help|-h)
      echo "usage: ./abliterate.sh [--preflight] [--skip-extract]"
      echo "  --preflight     host-side checks only (no GPU stages)"
      echo "  --skip-extract  reuse an existing data/out/text-lm (skip stage 0 extraction)"
      exit 0 ;;
    *) echo "warning: ignoring unknown arg: $arg" ;;
  esac
done

# ---- preflight (runs the checks and exits; never touches the GPU) ----
if [ "$DO_PREFLIGHT" = "1" ]; then
  preflight
fi

# ---- stage 0a: build the image if missing ----
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "building $IMAGE ..."
  docker build -t "$IMAGE" .
fi

# ---- stage 0b: model (download if missing) ----
if [ -f "$MODEL_DIR/config.json" ] && ls "$MODEL_DIR"/*.safetensors >/dev/null 2>&1; then
  echo "model found at $MODEL_DIR - skipping download"
else
  echo "downloading $MODEL_REPO -> $MODEL_DIR ..."
  python3 -c "from huggingface_hub import snapshot_download; \
    snapshot_download(repo_id='$MODEL_REPO', local_dir='$MODEL_DIR')"
fi

echo "====================================================================="
echo "QUICK ABLITERATION  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  model=$MODEL_NAME  prompts=$PROMPTS  configs=$CFGS"
echo "====================================================================="

# ---- stage 0: extract text-LM (prompt-independent; reuse if present) ----
if [ "$SKIP_EXTRACT" = "1" ]; then
  if [ -f "data/out/text-lm/COMPLETE" ]; then
    echo "(stage 0 skipped: --skip-extract, reusing data/out/text-lm)"
  else
    echo "FATAL: --skip-extract set but data/out/text-lm/COMPLETE is missing;"
    echo "       run once without --skip-extract to extract the text-LM first."
    exit 1
  fi
elif stage_done extract && [ -f "data/out/text-lm/COMPLETE" ]; then
  echo "(stage 0 skipped: text-lm already extracted)"
else
  echo; echo "### STAGE 0: extract text-LM from $MODEL_NAME"
  run_gpu /workspace/10_extract_text_lm.py
  mark_done extract
fi

# ---- stage 1: baseline ----
if stage_done baseline; then
  echo "(stage 1 skipped: baseline already done)"
else
  echo; echo "### STAGE 1: baseline (stock base, greedy)"
  run_gpu /tests/probe_checked.py \
    -e PROBE_MODEL_DIR=/out/text-lm -e PROBE_PROMPTS="$PROMPTS" \
    -e PROBE_THINKING=0 -e PROBE_MAX_TOKENS=250 -e PROBE_SANITY="$SANITY" \
    -e PROBE_DUMP=/out/baseline/probe_results.json
  python3 tests/gate.py baseline "$BASE_DUMP" 1
  mark_done baseline
fi

# ---- stage 2: mine ----
if stage_done mine; then
  echo "(stage 2 skipped: contrast already mined)"
else
  echo; echo "### STAGE 2: contrast mine (refusal vs benign, greedy)"
  run_gpu /workspace/73_mine_contrast.py -e CONTRAST_REFUSAL="$PROMPTS"
  python3 tests/gate.py contrast "$CONTRAST_JSON" 1 1
  mark_done mine
fi

# ---- stage 3: sweep ----
if stage_done sweep; then
  echo "(stage 3 skipped: sweep already done)"
else
  echo; echo "### STAGE 3: sweep (greedy)"
  run_gpu /tests/surgery_sweep.py \
    -e SWEEP_DIRECTION=/out/directions/contrast.pt -e SWEEP_CONFIGS="$CFGS" \
    -e SWEEP_PROMPTS="$PROMPTS" -e SWEEP_SANITY="$SANITY"
fi
# validate + derive the bake args from the (saved) sweep dump (fresh or resumed)
GATE_OUT=$(python3 tests/gate.py sweep "$SWEEP_DUMP" "$MAX_WINNER_REFUSED" || true)
echo "$GATE_OUT"
BAKE_LINE=$(printf '%s\n' "$GATE_OUT" | grep '^BAKE ARGS:' | sed 's/^BAKE ARGS: //' || true)
[ -n "$BAKE_LINE" ] || { echo "GATE FAIL: sweep produced no bake args"; exit 1; }
mark_done sweep
read -r DESIGN ALPHA KINDS <<< "$BAKE_LINE"
echo "winner: layers=$DESIGN alpha=$ALPHA kinds=$KINDS"

# ---- stage 4: bake (apply + merge) ----
if stage_done bake; then
  echo "(stage 4 skipped: bake already done)"
else
  echo; echo "### STAGE 4: bake ($DESIGN $KINDS @ $ALPHA)"
  run_gpu /workspace/40_apply.py \
    -e DIRECTION=/out/directions/contrast.pt -e SURGERY_LAYERS="$DESIGN" \
    -e SURGERY_KINDS="$KINDS" -e ALPHA="$ALPHA"
  run_gpu /workspace/50_merge.py
  mark_done bake
fi

# ---- stage 5: acceptance ----
if stage_done accept; then
  echo "(stage 5 skipped: acceptance already done)"
else
  echo; echo "### STAGE 5: acceptance (abliterated model, greedy)"
  run_gpu /tests/probe_checked.py \
    -e PROBE_MODEL_DIR="$ABL_OUT" -e PROBE_PROMPTS="$PROMPTS" \
    -e PROBE_THINKING=0 -e PROBE_MAX_TOKENS=250 -e PROBE_SANITY="$SANITY" \
    -e PROBE_DUMP=/out/probe/probe_results.json
  python3 tests/gate.py accept "$POST_DUMP" "$BASE_DUMP"
  mark_done accept
fi

echo
echo "### DONE - abliterated checkpoint at data/out/${MODEL_NAME}-abliterated"
