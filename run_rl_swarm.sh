#!/bin/bash
set -euo pipefail

ROOT=$PWD
ROOT_DIR="$(cd $(dirname ${BASH_SOURCE[0]}) && pwd)"
MODAL_PATH="$ROOT/modal-login"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"

export PUB_MULTI_ADDRS=""
export PEER_MULTI_ADDRS="/ip4/38.101.215.13/tcp/30002/p2p/QmQ2gEXoPJg6iMBSUFWGzAabS2VhnzuS782Y637hGjfsRJ"
export HOST_MULTI_ADDRS="/ip4/0.0.0.0/tcp/38331"
export IDENTITY_PATH="$ROOT/swarm.pem"
export HF_HUB_DOWNLOAD_TIMEOUT=120
ORG_ID=""
CPU_ONLY=${CPU_ONLY:-""}

GREEN_TEXT="\033[32m"
BLUE_TEXT="\033[34m"
RED_TEXT="\033[31m"
RESET_TEXT="\033[0m"
echo_green() { echo -e "$GREEN_TEXT$1$RESET_TEXT"; }
echo_blue()  { echo -e "$BLUE_TEXT$1$RESET_TEXT"; }
echo_red()   { echo -e "$RED_TEXT$1$RESET_TEXT"; }

cleanup() {
    echo_green ">> 正在关闭训练器..."
    rm -rf "$MODAL_PATH/temp-data/"*.json 2>/dev/null || true
    kill -- -$$ || true
    exit 0
}
trap cleanup EXIT
trap 'echo_red ">> 脚本执行时发生错误，请查看日志目录 logs/"' ERR

echo -e "\033[38;5;224m"
cat << "EOF"
    ██████  ██            ███████ ██     ██  █████  ██████  ███    ███
    ██   ██ ██            ██      ██     ██ ██   ██ ██   ██ ████  ████
    ██████  ██      █████ ███████ ██  █  ██ ███████ ██████  ██ ████ ██
    ██   ██ ██                 ██ ██ ███ ██ ██   ██ ██   ██ ██  ██  ██
    ██   ██ ███████       ███████  ███ ███  ██   ██ ██   ██ ██      ██
EOF

# === 非交互参数设定 ===
CONNECT_TO_TESTNET=true
USE_BIG_SWARM=false
PARAM_B=0.5
HUGGINGFACE_ACCESS_TOKEN="None"

SMALL_SWARM_CONTRACT="0x69C6e1D608ec64885E7b185d39b04B491a71768C"
BIG_SWARM_CONTRACT="0x6947c6E196a48B77eFa9331EC1E3e45f3Ee5Fd58"
SWARM_CONTRACT="$([[ "$USE_BIG_SWARM" = true ]] && echo "$BIG_SWARM_CONTRACT" || echo "$SMALL_SWARM_CONTRACT")"

# === 启动 modal-login 登录页 ===
if [ "$CONNECT_TO_TESTNET" = true ]; then
    echo_blue "🔐 登录以创建以太坊服务器钱包账户..."

    cd "$MODAL_PATH" || { echo_red "❌ modal-login 目录不存在：$MODAL_PATH"; exit 1; }

    if ! command -v node >/dev/null; then
        echo_blue "安装 Node.js 环境中..."
        export NVM_DIR="$HOME/.nvm"
        [ ! -d "$NVM_DIR" ] && curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
        [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
        nvm install node
    fi

    if ! command -v yarn >/dev/null; then
        echo_blue "安装 Yarn..."
        npm install -g --silent yarn
    fi

    ENV_FILE="$MODAL_PATH/.env"
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' "3s/.*/SMART_CONTRACT_ADDRESS=$SWARM_CONTRACT/" "$ENV_FILE"
    else
        sed -i "3s/.*/SMART_CONTRACT_ADDRESS=$SWARM_CONTRACT/" "$ENV_FILE"
    fi

    echo_blue "🔧 构建 modal-login 前端页面中..."
    # 自动安装依赖（兼容 Yarn v2/v3/v4，避免锁文件报错）
    yarn config set enableImmutableInstalls false
    yarn install
    yarn build >> "$LOG_DIR/yarn.log" 2>&1

    echo_blue "🚀 启动 modal-login 服务..."
    yarn start >> "$LOG_DIR/yarn.log" 2>&1 &
    sleep 2

    open http://localhost:3000 2>/dev/null || echo "⚠️ 请手动打开浏览器访问 http://localhost:3000"

    cd "$ROOT"

    echo_green "⌛ 正在等待用户登录信息生成（userData.json）..."
    while [ ! -f "$MODAL_PATH/temp-data/userData.json" ]; do sleep 2; done

    echo_green "✅ 已获取 userData.json，继续执行后续步骤..."
    ORG_ID=$(awk 'BEGIN { FS = "\"" } !/^[ \t]*[{}]/ { print $(NF - 1); exit }' "$MODAL_PATH/temp-data/userData.json")
    echo_green "✅ 已提取 ORG_ID：$ORG_ID"

    echo_blue "🔑 等待 API 密钥激活中..."
    while true; do
        STATUS=$(curl -s "http://localhost:3000/api/get-api-key-status?orgId=$ORG_ID")
        echo "🔍 当前状态：$STATUS"
        [[ "$STATUS" == "activated" ]] && break || sleep 2
    done

    echo_green "🔓 API 密钥已激活，准备开始训练..."
fi

# === 安装依赖并启动训练器 ===
echo_green "📦 安装训练所需依赖..."
pip install --upgrade pip

if [ -n "$CPU_ONLY" ] || ! command -v nvidia-smi &> /dev/null; then
    pip install -r "$ROOT/requirements-cpu.txt"
    CONFIG_PATH="$ROOT/hivemind_exp/configs/mac/grpo-qwen-2.5-0.5b-deepseek-r1.yaml"
    GAME="gsm8k"
else
    pip install -r "$ROOT/requirements-gpu.txt"
    pip install flash-attn --no-build-isolation
    case "$PARAM_B" in
        32 | 72) CONFIG_PATH="$ROOT/hivemind_exp/configs/gpu/grpo-qwen-2.5-${PARAM_B}b-bnb-4bit-deepseek-r1.yaml" ;;
        *)       CONFIG_PATH="$ROOT/hivemind_exp/configs/gpu/grpo-qwen-2.5-${PARAM_B}b-deepseek-r1.yaml" ;;
    esac
    GAME="gsm8k"
fi

# =====（可选）替换有问题的 accelerate/data_loader.py =====
ACC_FILE="$PWD/.venv/lib/python3.13/site-packages/accelerate/data_loader.py"
rm "$ACC_FILE"
cp "$PWD/data_loader.py" "$ACC_FILE"

echo_green "🚀 启动训练器..."
if [ -n "$ORG_ID" ]; then
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
        --public_maddr "$PUB_MULTI_ADDRS" \
        --initial_peers "$PEER_MULTI_ADDRS" \
        --host_maddr "$HOST_MULTI_ADDRS" \
        --config "$CONFIG_PATH" \
        --game "$GAME"
fi

wait
