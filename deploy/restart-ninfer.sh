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
MODE_FILE=${NINFER_MODE_FILE:-$PID_FILE.mode}
PORT=${NINFER_PORT:-8321}
# Fixed shared key for this LAN deployment; reused across restarts.
API_KEY='sk-may-the-tokens-be-with-you'
MAX_CONTEXT=${NINFER_MAX_CONTEXT:-245760}
KV_CAPACITY=${NINFER_KV_CAPACITY:-245760}
DEFAULT_MAX_TOKENS=${NINFER_DEFAULT_MAX_TOKENS:-32768}
PREFILL_CHUNK=${NINFER_PREFILL_CHUNK:-1024}
DRAFT_TOKENS=${NINFER_DRAFT_TOKENS:-3}
TOOL_REPLAY_CACHE_MIB=${NINFER_TOOL_REPLAY_CACHE_MIB:-1024}
state_cache_default_dir() {
    # Keep retained snapshots on the dedicated data volume when it is usable. The bundle-local
    # directory remains a portable fallback for development and hosts without /data.
    local preferred=/data/ninfer-state-cache
    local fallback=$BUNDLE_ROOT/state-cache
    if [[ -n ${NINFER_STATE_CACHE_DIR:-} ]]; then
        printf '%s\n' "$NINFER_STATE_CACHE_DIR"
        return
    fi
    if [[ -d /data && -r /data && -x /data ]]; then
        if [[ ! -d "$preferred" ]]; then
            mkdir -p -- "$preferred" 2>/dev/null || true
        fi
        if [[ -d "$preferred" && -w "$preferred" && -x "$preferred" ]]; then
            printf '%s\n' "$preferred"
            return
        fi
    fi
    printf '%s\n' "$fallback"
}

STATE_CACHE_DIR=$(state_cache_default_dir)
STATE_CACHE_MAX_MIB=${NINFER_STATE_CACHE_MAX_MIB:-51200}
STATE_CACHE_RAM_MIB=${NINFER_STATE_CACHE_RAM_MIB:-8192}
STATE_CACHE_IDLE_MS=${NINFER_STATE_CACHE_IDLE_MS:-1000}
TURBO_POWER_LIMIT_W=280
FAN_CURVE_SERVICE=${NINFER_FAN_CURVE_SERVICE:-nvidia-fan-curve.service}
TURBO_MODE=false
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

gpu_power_limit() {
    local field=$1
    nvidia-smi -i 0 --query-gpu="$field" --format=csv,noheader,nounits 2>/dev/null |
        head -n 1 | tr -d '[:space:]'
}

restore_default_power_limit() {
    local default_limit current_limit
    default_limit=$(gpu_power_limit power.default_limit)
    current_limit=$(gpu_power_limit power.limit)
    [[ "$default_limit" =~ ^[0-9]+([.][0-9]+)?$ ]] || {
        echo "Unable to read GPU 0 default power limit" >&2
        return 1
    }
    if [[ "$current_limit" == "$default_limit" ]]; then
        return 0
    fi
    printf 'Restoring GPU 0 default power limit: %s W\n' "$default_limit"
    sudo nvidia-smi -i 0 --power-limit="$default_limit"
}

enable_aggressive_fan_curve() {
    if ! systemctl is-active --quiet "$FAN_CURVE_SERVICE"; then
        printf 'Starting aggressive GPU fan curve: %s\n' "$FAN_CURVE_SERVICE"
        sudo systemctl start "$FAN_CURVE_SERVICE" || return 1
    fi
    systemctl is-active --quiet "$FAN_CURVE_SERVICE" || {
        echo "Aggressive GPU fan curve is not active: $FAN_CURVE_SERVICE" >&2
        return 1
    }
}

stop_server() {
    local pids power_restore_status=0
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
    rm -f "$PID_FILE" "$MODE_FILE"
    restore_default_power_limit || power_restore_status=$?

    for _ in $(seq 1 60); do
        local used
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null |
            head -n 1 | tr -d ' ')
        [[ ${used:-0} -lt 1000 ]] && return "$power_restore_status"
        sleep 1
    done
    echo "GPU memory did not fall below 1000 MiB" >&2
    return 1
}

start_server() {
    [[ -x "$BIN" ]] || { echo "Missing executable: $BIN" >&2; return 1; }
    [[ -f "$MODEL" ]] || { echo "Missing model: $MODEL" >&2; return 1; }
    rm -f "$MODE_FILE"
    local -a sampling_args=()
    if [[ -n ${NINFER_SEED:-} ]]; then
        sampling_args+=(--seed "$NINFER_SEED")
    fi

    if [[ "$TURBO_MODE" == true ]]; then
        enable_aggressive_fan_curve || return 1
        GPU_POWER_LIMIT_W=$TURBO_POWER_LIMIT_W
        printf 'Turbo mode: %s W power limit | aggressive fan curve active\n' \
            "$GPU_POWER_LIMIT_W"
    fi

    if [[ -n "$GPU_POWER_LIMIT_W" ]]; then
        [[ "$GPU_POWER_LIMIT_W" =~ ^[1-9][0-9]*$ ]] || {
            echo "NINFER_GPU_POWER_LIMIT_W must be a positive integer in watts" >&2
            return 1
        }
        printf 'Applying requested GPU power limit: %s W\n' "$GPU_POWER_LIMIT_W"
        sudo nvidia-smi -i 0 --power-limit="$GPU_POWER_LIMIT_W" || return 1
    fi

    printf 'State cache directory: %s\n' "$STATE_CACHE_DIR"

    nohup setsid env LD_LIBRARY_PATH="$RUNTIME_LIBRARY_PATH" "$BIN" "$MODEL" \
        --host 0.0.0.0 --port "$PORT" --api-key "$API_KEY" \
        --max-context "$MAX_CONTEXT" --kv-capacity "$KV_CAPACITY" --kv-dtype kvarn \
        --prefill-chunk "$PREFILL_CHUNK" \
        --spec mtp --draft-tokens "$DRAFT_TOKENS" --lm-head-draft \
        --max-concurrency 2 \
        --pending-timeout-ms 600000 \
        --default-max-tokens "$DEFAULT_MAX_TOKENS" \
        --tool-replay-cache-mib "$TOOL_REPLAY_CACHE_MIB" \
        --state-cache-dir "$STATE_CACHE_DIR" --state-cache-max-mib "$STATE_CACHE_MAX_MIB" \
        --state-cache-ram-mib "$STATE_CACHE_RAM_MIB" \
        --state-cache-idle-ms "$STATE_CACHE_IDLE_MS" \
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
            rm -f "$MODE_FILE"
            restore_default_power_limit || true
            return 1
        fi
        if curl -fsS -H "Authorization: Bearer $API_KEY" \
            "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
            printf '%sReady%s: http://0.0.0.0:%s/v1' "$COLOR_OK" "$COLOR_RESET" "$PORT"
            if [[ -n "$API_KEY" ]]; then
                printf ' | API Key: %s' "$API_KEY"
            fi
            printf '\n'
            if [[ "$TURBO_MODE" == true ]]; then
                printf 'turbo\n' >"$MODE_FILE"
            else
                printf 'standard\n' >"$MODE_FILE"
            fi
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
    kill "$pid" 2>/dev/null || true
    rm -f "$MODE_FILE"
    restore_default_power_limit || true
    return 1
}

watch_metrics() {
    [[ -f "$LOG" ]] || { echo "Missing log: $LOG" >&2; return 1; }
    local pids
    local -a watch_args=()
    pids=$(server_pids)
    [[ -z "$pids" ]] || watch_args+=(--pid "${pids%%$'\n'*}")
    if [[ -f "$MODE_FILE" && "$(<"$MODE_FILE")" == turbo ]]; then
        watch_args+=(--turbo)
    fi
    python3 "$SOURCE_ROOT/deploy/watch_ninfer.py" --log "$LOG" --max-concurrency 2 \
        --pid-file "$PID_FILE" \
        --server-url "http://0.0.0.0:$PORT/v1" --api-key "$API_KEY" --model "$MODEL" \
        --max-context "$MAX_CONTEXT" --kv-capacity "$KV_CAPACITY" \
        --max-output "$DEFAULT_MAX_TOKENS" --draft-tokens "$DRAFT_TOKENS" \
        --prefill-chunk "$PREFILL_CHUNK" \
        --tool-replay-cache-mib "$TOOL_REPLAY_CACHE_MIB" \
        --state-cache-dir "$STATE_CACHE_DIR" --state-cache-max-mib "$STATE_CACHE_MAX_MIB" \
        --state-cache-ram-mib "$STATE_CACHE_RAM_MIB" \
        --state-cache-idle-ms "$STATE_CACHE_IDLE_MS" \
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

NINFER_ACTION=restart
if [[ $# -gt 0 && $1 != "--turbo" ]]; then
    NINFER_ACTION=$1
    shift
fi
WATCH_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --turbo)
            TURBO_MODE=true
            shift
            ;;
        --tool-replay-cache-mib)
            [[ $# -ge 2 ]] || { echo "--tool-replay-cache-mib needs a value" >&2; exit 2; }
            TOOL_REPLAY_CACHE_MIB=$2
            shift 2
            ;;
        --state-cache-dir)
            [[ $# -ge 2 ]] || { echo "--state-cache-dir needs a value" >&2; exit 2; }
            STATE_CACHE_DIR=$2
            shift 2
            ;;
        --state-cache-max-mib)
            [[ $# -ge 2 ]] || { echo "--state-cache-max-mib needs a value" >&2; exit 2; }
            STATE_CACHE_MAX_MIB=$2
            shift 2
            ;;
        --state-cache-ram-mib)
            [[ $# -ge 2 ]] || { echo "--state-cache-ram-mib needs a value" >&2; exit 2; }
            STATE_CACHE_RAM_MIB=$2
            shift 2
            ;;
        --state-cache-idle-ms)
            [[ $# -ge 2 ]] || { echo "--state-cache-idle-ms needs a value" >&2; exit 2; }
            STATE_CACHE_IDLE_MS=$2
            shift 2
            ;;
        *) WATCH_ARGS+=("$1"); shift ;;
    esac
done

case $NINFER_ACTION in
    restart|restart-daemon|--daemon|start) ;;
    *)
        if [[ "$TURBO_MODE" == true ]]; then
            echo "--turbo is only valid with restart, restart-daemon, --daemon, or start" >&2
            exit 2
        fi
        ;;
esac
[[ "$TOOL_REPLAY_CACHE_MIB" =~ ^[0-9]+$ ]] || {
    echo "--tool-replay-cache-mib must be a nonnegative integer in MiB" >&2
    exit 2
}
[[ "$STATE_CACHE_MAX_MIB" =~ ^[0-9]+$ ]] || {
    echo "--state-cache-max-mib must be a nonnegative integer in MiB" >&2
    exit 2
}
[[ "$STATE_CACHE_RAM_MIB" =~ ^[0-9]+$ && "$STATE_CACHE_RAM_MIB" -gt 0 ]] || {
    echo "--state-cache-ram-mib must be positive in MiB" >&2
    exit 2
}
[[ "$STATE_CACHE_IDLE_MS" =~ ^[0-9]+$ ]] || {
    echo "--state-cache-idle-ms must be a nonnegative integer" >&2
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
        if [[ -n "$(server_pids)" ]]; then
            if [[ "$TURBO_MODE" == true ]]; then
                echo "ninfer-serve is already running; use restart --turbo to change mode" >&2
                exit 2
            fi
            show_status
            exit 0
        fi
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
        echo "Usage: $0 [restart|restart-daemon|start|stop|status|watch|logs] [--turbo] [--state-cache-dir DIR] [--state-cache-max-mib N] [--state-cache-idle-ms N] [--tool-replay-cache-mib N] [--details] [--once] [--color auto|always|never]" >&2
        exit 2
        ;;
esac
