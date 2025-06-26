#!/bin/bash
set -euo pipefail

# ===== 基本设置 =====
ROOT=$PWD
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export IDENTITY_PATH="$ROOT/swarm.pem"
export HF_HUB_DOWNLOAD_TIMEOUT=120
export TOKENIZERS_PARALLELISM=false

USE_BIG_SWARM=false           # 默认加入 Math swarm（非 Hard）
PARAM_B=0.5                   # 默认 0.5B 模型
CONNECT_TO_TESTNET=true       # 默认连接 testnet
HUGGINGFACE_ACCESS_TOKEN="None"

SMALL_SWARM_CONTRACT="0x69C6e1D608ec64885E7b185d39b04B491a71768C"
BIG_SWARM_CONTRACT="0x6947c6E196a48B77eFa9331EC1E3e45f3Ee5Fd58"
SWARM_CONTRACT="$SMALL_SWARM_CONTRACT"

DEFAULT_PEER_MULTI_ADDRS="/ip4/0.0.0.0/tcp/38331"
export HOST_MULTI_ADDRS=${HOST_MULTI_ADDRS:-$DEFAULT_PEER_MULTI_ADDRS}

GREEN="\033[32m"
RED="\033[31m"
BLUE="\033[34m"
RESET="\033[0m"
echo_green() { echo -e "$GREEN$1$RESET"; }
echo_red()   { echo -e "$RED$1$RESET"; }
echo_blue()  { echo -e "$BLUE$1$RESET"; }

cleanup() {
    echo_green "🧹 正在关闭训练器..."
    pkill -f "yarn start" || true
    rm -rf "$ROOT/modal-login/temp-data/"*.json 2>/dev/null || true
    kill -- -$$ || true
    exit 0
}
errnotify() {
    echo_red "❌ 出错啦，请查看 $ROOT/logs 获取详细日志"
}
trap cleanup EXIT
trap errnotify ERR

echo -e "\033[38;5;224m"
cat << "EOF"
    ██████  ██            ███████ ██     ██  █████  ██████  ███    ███
    ██   ██ ██            ██      ██     ██ ██   ██ ██   ██ ████  ████
    ██████  ██      █████ ███████ ██  █  ██ ███████ ██████  ██ ████ ██
    ██   ██ ██                 ██ ██ ███ ██ ██   ██ ██   ██ ██  ██  ██
    ██   ██ ███████       ███████  ███ ███  ██   ██ ██   ██ ██      ██

                  Gensyn RL Swarm 启动器
EOF
echo -e "\033[0m"


# ===== Python 模块检测器 =====
ensure_python_package() {
    python3 -c "import $1" 2>/dev/null || {
        echo_blue "📦 安装 Python 模块：$1"
        pip install "$1"
    }
}
ensure_python_package torch
ensure_python_package psutil

# ===== 检查 Yarn =====
if ! command -v yarn &> /dev/null; then
    echo_blue "📦 安装 Yarn..."
    npm install -g yarn
fi

# ===== 启动 modal-login 登录页 =====
if [ "$CONNECT_TO_TESTNET" = true ]; then
    cd modal-login
    sed -i.bak "3s/.*/SMART_CONTRACT_ADDRESS=$SWARM_CONTRACT/" .env
    yarn install --immutable
    echo "🚀 启动登录服务器..."
    yarn build >> "$ROOT/logs/yarn.log" 2>&1
    yarn start >> "$ROOT/logs/yarn.log" 2>&1 &

    sleep 5
    open http://localhost:3000 2>/dev/null || echo "⚠️ 请手动访问 http://localhost:3000 登录"
    cd ..

    echo_green "⌛ 等待用户完成登录..."
    while [ ! -f "modal-login/temp-data/userData.json" ]; do sleep 3; done
    ORG_ID=$(awk 'BEGIN { FS = "\"" } !/^[ \t]*[{}]/ { print $(NF - 1); exit }' modal-login/temp-data/userData.json)
    echo_green "✅ ORG_ID = $ORG_ID"

    while true; do
        STATUS=$(curl -s "http://localhost:3000/api/get-api-key-status?orgId=$ORG_ID")
        [[ "$STATUS" == "activated" ]] && break || sleep 2
    done
    echo_green "🔓 API Key 已激活，准备开始训练..."
fi

# ===== 安装训练依赖 =====
mkdir -p "$ROOT/logs"
pip install --upgrade pip
if ! command -v nvidia-smi &> /dev/null; then
    pip install -r "$ROOT/requirements-cpu.txt"
    CONFIG_PATH="$ROOT/hivemind_exp/configs/mac/grpo-qwen-2.5-0.5b-deepseek-r1.yaml"
    GAME="gsm8k"
else
    pip install -r "$ROOT/requirements-gpu.txt"
    pip install flash-attn --no-build-isolation
    case "$PARAM_B" in
        32|72) CONFIG_PATH="$ROOT/hivemind_exp/configs/gpu/grpo-qwen-2.5-${PARAM_B}b-bnb-4bit-deepseek-r1.yaml" ;;
        *) CONFIG_PATH="$ROOT/hivemind_exp/configs/gpu/grpo-qwen-2.5-${PARAM_B}b-deepseek-r1.yaml" ;;
    esac
    GAME=$([ "$USE_BIG_SWARM" = true ] && echo "dapo" || echo "gsm8k")
fi

# ===== 启动训练 =====
echo_green "🎯 启动 RL Swarm 训练任务..."
if [ -n "${ORG_ID:-}" ]; then
    python -m hivemind_exp.gsm8k.train_single_gpu \
        --hf_token "$HUGGINGFACE_ACCESS_TOKEN" \
        --identity_path "$IDENTITY_PATH" \
        --modal_org_id "$ORG_ID" \
        --contract_address "$SWARM_CONTRACT" \
        --config "$CONFIG_PATH" \
        --game "$GAME"
else
    python -m hivemind_exp.gsm8k.train_single_gpu \
        --hf_token "$HUGGINGFACE_ACCESS_TOKEN" \
        --identity_path "$IDENTITY_PATH" \
        --config "$CONFIG_PATH" \
        --game "$GAME"
fi

wait
