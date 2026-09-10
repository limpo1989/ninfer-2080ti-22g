#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [[ -d "$SCRIPT_DIR/ninfer-src" ]]; then
    BUNDLE_ROOT=$SCRIPT_DIR
    SOURCE_ROOT=$BUNDLE_ROOT/ninfer-src
else
    SOURCE_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
    BUNDLE_ROOT=$(CDPATH= cd -- "$SOURCE_ROOT/.." && pwd)
fi

BIN=${NINFER_BIN:-$SOURCE_ROOT/build/apps/ninfer-serve}
MODEL=${NINFER_MODEL:-$BUNDLE_ROOT/models/qwen3_8_27b.ninfer}
LOG=${NINFER_LOG:-$BUNDLE_ROOT/serve.log}
PID_FILE=${NINFER_PID_FILE:-$BUNDLE_ROOT/ninfer-serve.pid}
PORT=${NINFER_PORT:-8321}
# Fixed shared key for this LAN deployment; reused across restarts.
API_KEY='sk-may-the-tokens-be-with-you'
MAX_CONTEXT=${NINFER_MAX_CONTEXT:-245760}
KV_CAPACITY=${NINFER_KV_CAPACITY:-245760}
DEFAULT_MAX_TOKENS=${NINFER_DEFAULT_MAX_TOKENS:-131072}
PREFILL_CHUNK=${NINFER_PREFILL_CHUNK:-1024}
DRAFT_TOKENS=${NINFER_DRAFT_TOKENS:-3}
TOOL_REPLAY_CACHE_MIB=${NINFER_TOOL_REPLAY_CACHE_MIB:-1024}
# Optional power override, disabled by default. Example: NINFER_GPU_POWER_LIMIT_W=280
# An empty value leaves the driver's current limit unchanged (250 W by default on this host).
GPU_POWER_LIMIT_W=${NINFER_GPU_POWER_LIMIT_W:-}
DEPS_PREFIX=${NINFER_DEPS_PREFIX:-$HOME/.local/ninfer-deps}

export PATH=/usr/local/cuda-12.8/bin:/usr/local/cuda/bin:$PATH
RUNTIME_LIBRARY_PATH=$DEPS_PREFIX/lib:/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}

if [[ -t 1 ]]; then
    COLOR_TITLE=$'\033[1;36m'
    COLOR_OK=$'\033[1;32m'
    COLOR_DIM=$'\033[2m'
    COLOR_RESET=$'\033[0m'
else
    COLOR_TITLE=
    COLOR_OK=
    COLOR_DIM=
    COLOR_RESET=
fi

server_pids() {
    local pid actual expected
    expected=$(readlink -f -- "$BIN") || return 0
    while IFS= read -r pid; do
        actual=$(readlink -- "/proc/$pid/exe" 2>/dev/null || true)
        actual=${actual%" (deleted)"}
        [[ "$actual" == "$expected" ]] && printf '%s\n' "$pid"
    done < <(pgrep -f -- "$BIN" || true)
    return 0
}

stop_server() {
    local pids
    pids=$(server_pids)
    if [[ -n "$pids" ]]; then
        echo "Stopping ninfer-serve: $pids"
        kill $pids 2>/dev/null || true
        for _ in $(seq 1 30); do
            [[ -z "$(server_pids)" ]] && break
            sleep 1
        done
        pids=$(server_pids)
        if [[ -n "$pids" ]]; then
            echo "Force stopping ninfer-serve: $pids"
            kill -9 $pids 2>/dev/null || true
        fi
    fi
    rm -f "$PID_FILE"

    for _ in $(seq 1 60); do
        local used
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null |
            head -n 1 | tr -d ' ')
        [[ ${used:-0} -lt 1000 ]] && return
        sleep 1
    done
    echo "GPU memory did not fall below 1000 MiB" >&2
    return 1
}

start_server() {
    [[ -x "$BIN" ]] || { echo "Missing executable: $BIN" >&2; return 1; }
    [[ -f "$MODEL" ]] || { echo "Missing model: $MODEL" >&2; return 1; }
    local -a sampling_args=()
    if [[ -n ${NINFER_SEED:-} ]]; then
        sampling_args+=(--seed "$NINFER_SEED")
    fi

    if [[ -n "$GPU_POWER_LIMIT_W" ]]; then
        [[ "$GPU_POWER_LIMIT_W" =~ ^[1-9][0-9]*$ ]] || {
            echo "NINFER_GPU_POWER_LIMIT_W must be a positive integer in watts" >&2
            return 1
        }
        printf 'Applying requested GPU power limit: %s W\n' "$GPU_POWER_LIMIT_W"
        sudo nvidia-smi -i 0 --power-limit="$GPU_POWER_LIMIT_W" || return 1
    fi

    nohup setsid env LD_LIBRARY_PATH="$RUNTIME_LIBRARY_PATH" "$BIN" "$MODEL" \
        --host 0.0.0.0 --port "$PORT" --api-key "$API_KEY" \
        --max-context "$MAX_CONTEXT" --kv-capacity "$KV_CAPACITY" --kv-dtype kvarn \
        --prefill-chunk "$PREFILL_CHUNK" \
        --spec mtp --draft-tokens "$DRAFT_TOKENS" --lm-head-draft \
        --max-concurrency 2 \
        --pending-timeout-ms 600000 \
        --default-max-tokens "$DEFAULT_MAX_TOKENS" \
        --tool-replay-cache-mib "$TOOL_REPLAY_CACHE_MIB" \
        --temperature 0.6 --presence-penalty 1.0 \
        "${sampling_args[@]}" \
        >"$LOG" 2>&1 </dev/null &
    local pid=$!
    echo "$pid" >"$PID_FILE"
    printf '%sStarting ninfer-serve%s (pid %s); waiting for model load...\n' \
        "$COLOR_TITLE" "$COLOR_RESET" "$pid"

    for _ in $(seq 1 180); do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "ninfer-serve exited during startup" >&2
            tail -n 30 "$LOG" >&2 || true
            return 1
        fi
        if curl -fsS -H "Authorization: Bearer $API_KEY" \
            "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
            printf '%sReady%s: http://0.0.0.0:%s/v1' "$COLOR_OK" "$COLOR_RESET" "$PORT"
            if [[ -n "$API_KEY" ]]; then
                printf ' | API Key: %s' "$API_KEY"
            fi
            printf '\n'
            printf 'Context: %s | shared KV: %s | default max_tokens: %s\n' \
                "$MAX_CONTEXT" "$KV_CAPACITY" "$DEFAULT_MAX_TOKENS"
            nvidia-smi --query-gpu=memory.used,memory.total,temperature.gpu,fan.speed \
                --format=csv,noheader
            return
        fi
        sleep 1
    done
    echo "Timed out waiting for http://127.0.0.1:$PORT/v1/models" >&2
    tail -n 30 "$LOG" >&2 || true
    return 1
}

watch_metrics() {
    [[ -f "$LOG" ]] || { echo "Missing log: $LOG" >&2; return 1; }
    local pids
    local -a watch_args=()
    pids=$(server_pids)
    [[ -z "$pids" ]] || watch_args+=(--pid "${pids%%$'\n'*}")
    python3 "$SOURCE_ROOT/deploy/watch_ninfer.py" --log "$LOG" --max-concurrency 2 \
        --pid-file "$PID_FILE" \
        --server-url "http://0.0.0.0:$PORT/v1" --api-key "$API_KEY" --model "$MODEL" \
        --max-context "$MAX_CONTEXT" --kv-capacity "$KV_CAPACITY" \
        --max-output "$DEFAULT_MAX_TOKENS" --draft-tokens "$DRAFT_TOKENS" \
        --prefill-chunk "$PREFILL_CHUNK" \
        --tool-replay-cache-mib "$TOOL_REPLAY_CACHE_MIB" \
        "${watch_args[@]}" "$@"
}

show_status() {
    local pids
    pids=$(server_pids)
    if [[ -n "$pids" ]]; then
        echo "ninfer-serve running: $pids"
        curl -fsS -H "Authorization: Bearer $API_KEY" "http://127.0.0.1:$PORT/v1/models"
        echo
        nvidia-smi --query-gpu=memory.used,memory.total,temperature.gpu,fan.speed \
            --format=csv,noheader
    else
        echo "ninfer-serve is stopped"
        return 1
    fi
}

NINFER_ACTION=${1:-restart}
[[ $# -eq 0 ]] || shift
WATCH_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --tool-replay-cache-mib)
            [[ $# -ge 2 ]] || { echo "--tool-replay-cache-mib needs a value" >&2; exit 2; }
            TOOL_REPLAY_CACHE_MIB=$2
            shift 2
            ;;
        *) WATCH_ARGS+=("$1"); shift ;;
    esac
done
[[ "$TOOL_REPLAY_CACHE_MIB" =~ ^[0-9]+$ ]] || {
    echo "--tool-replay-cache-mib must be a nonnegative integer in MiB" >&2
    exit 2
}

case $NINFER_ACTION in
    restart)
        stop_server
        start_server
        watch_metrics "${WATCH_ARGS[@]}"
        ;;
    restart-daemon|--daemon)
        stop_server
        start_server
        ;;
    start)
        [[ -z "$(server_pids)" ]] || { show_status; exit 0; }
        start_server
        ;;
    stop|--stop)
        stop_server
        echo "ninfer-serve stopped"
        ;;
    status)
        show_status
        ;;
    watch|--watch)
        watch_metrics "${WATCH_ARGS[@]}"
        ;;
    logs)
        tail -n 100 -F "$LOG"
        ;;
    *)
        echo "Usage: $0 [restart|restart-daemon|start|stop|status|watch|logs] [--tool-replay-cache-mib N] [--details] [--once] [--color auto|always|never]" >&2
        exit 2
        ;;
esac
