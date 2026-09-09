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
MAX_CONTEXT=${NINFER_MAX_CONTEXT:-245760}
KV_CAPACITY=${NINFER_KV_CAPACITY:-245760}
DEFAULT_MAX_TOKENS=${NINFER_DEFAULT_MAX_TOKENS:-131072}
PREFILL_CHUNK=${NINFER_PREFILL_CHUNK:-1024}
DRAFT_TOKENS=${NINFER_DRAFT_TOKENS:-3}
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

print_rule() {
    printf '%s\n' '+----------+----------+------------+------------+----------+-----------+---------+---------+'
}

server_pids() {
    pgrep -f -- "$BIN" || true
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

    if [[ -n "$GPU_POWER_LIMIT_W" ]]; then
        [[ "$GPU_POWER_LIMIT_W" =~ ^[1-9][0-9]*$ ]] || {
            echo "NINFER_GPU_POWER_LIMIT_W must be a positive integer in watts" >&2
            return 1
        }
        printf 'Applying requested GPU power limit: %s W\n' "$GPU_POWER_LIMIT_W"
        sudo nvidia-smi -i 0 --power-limit="$GPU_POWER_LIMIT_W" || return 1
    fi

    nohup env LD_LIBRARY_PATH="$RUNTIME_LIBRARY_PATH" "$BIN" "$MODEL" \
        --host 0.0.0.0 --port "$PORT" \
        --max-context "$MAX_CONTEXT" --kv-capacity "$KV_CAPACITY" --kv-dtype kvarn \
        --prefill-chunk "$PREFILL_CHUNK" \
        --spec mtp --draft-tokens "$DRAFT_TOKENS" --lm-head-draft \
        --max-concurrency 2 \
        --pending-timeout-ms 600000 \
        --default-max-tokens "$DEFAULT_MAX_TOKENS" \
        --temperature 0.6 --presence-penalty 1.0 \
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
        if curl -fsS "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
            printf '%sReady%s: http://0.0.0.0:%s/v1\n' "$COLOR_OK" "$COLOR_RESET" "$PORT"
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

    printf '\n%sNInfer live request metrics%s\n' "$COLOR_TITLE" "$COLOR_RESET"
    printf '%sCtrl+C exits this view; the HTTP service keeps running.%s\n' \
        "$COLOR_DIM" "$COLOR_RESET"
    print_rule
    printf '| %-8s | %-8s | %10s | %10s | %8s | %9s | %7s | %7s |\n' \
        'Time' 'TTFB' 'Prefill' 'Decode' 'Output' 'Cache hit' 'MTP' 'Wall'
    print_rule

    trap 'printf "\nMonitoring stopped; ninfer-serve is still running.\n"' INT
    tail -n 20 -F "$LOG" 2>/dev/null | while IFS= read -r line; do
        [[ "$line" == *'] done '* ]] || continue

        local timestamp prompt generated cached ttft prefill decode wall mtp cache_hit
        timestamp=$(printf '%s\n' "$line" |
            grep -oE '[0-9]{2}:[0-9]{2}:[0-9]{2}' | head -n 1)
        prompt=$(printf '%s\n' "$line" | sed -nE 's/.* prompt=([0-9]+).*/\1/p')
        generated=$(printf '%s\n' "$line" | sed -nE 's/.* gen=([0-9]+).*/\1/p')
        cached=$(printf '%s\n' "$line" | sed -nE 's/.* cache=([0-9]+).*/\1/p')
        ttft=$(printf '%s\n' "$line" | sed -nE 's/.* ttft=([^ ]+).*/\1/p')
        prefill=$(printf '%s\n' "$line" | sed -nE 's/.* prefill=([^ ]+).*/\1/p')
        decode=$(printf '%s\n' "$line" | sed -nE 's/.* decode=([^ ]+).*/\1/p')
        wall=$(printf '%s\n' "$line" | sed -nE 's/.* wall=([^ ]+).*/\1/p')
        mtp=$(printf '%s\n' "$line" |
            sed -nE 's/.* speculative=[^ ]+ [^ ]+ \(([0-9.]+%)\).*/\1/p')

        [[ -n "$timestamp" ]] || timestamp='-'
        [[ -n "$ttft" ]] || ttft='-'
        [[ -n "$prefill" ]] || prefill='-'
        [[ -n "$decode" ]] || decode='-'
        [[ -n "$generated" ]] || generated='-'
        [[ -n "$wall" ]] || wall='-'
        [[ -n "$mtp" ]] || mtp='-'
        if [[ -n "$prompt" && -n "$cached" && "$prompt" -gt 0 ]]; then
            cache_hit=$((cached * 100 / prompt))%
        else
            cache_hit='-'
        fi

        printf '%s| %-8s | %-8s | %10s | %10s | %8s | %9s | %7s | %7s |%s\n' \
            "$COLOR_OK" "$timestamp" "$ttft" "$prefill" "$decode" "$generated" \
            "$cache_hit" "$mtp" "$wall" "$COLOR_RESET"
    done
    trap - INT
}

show_status() {
    local pids
    pids=$(server_pids)
    if [[ -n "$pids" ]]; then
        echo "ninfer-serve running: $pids"
        curl -fsS "http://127.0.0.1:$PORT/v1/models"
        echo
        nvidia-smi --query-gpu=memory.used,memory.total,temperature.gpu,fan.speed \
            --format=csv,noheader
    else
        echo "ninfer-serve is stopped"
        return 1
    fi
}

case ${1:-restart} in
    restart)
        stop_server
        start_server
        watch_metrics
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
        show_status >/dev/null
        watch_metrics
        ;;
    logs)
        tail -n 100 -F "$LOG"
        ;;
    *)
        echo "Usage: $0 [restart|restart-daemon|start|stop|status|watch|logs]" >&2
        exit 2
        ;;
esac
