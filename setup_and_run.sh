#!/bin/bash
set -euo pipefail

# ===== 配置输出颜色 =====
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

log() { echo -e "${BLUE}➤ $1${NC}"; }
success() { echo -e "${GREEN}✔ $1${NC}"; }
error() { echo -e "${RED}✖ $1${NC}"; }

# ===== 项目目录定位 =====
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ===== 设置 MPS 参数（macOS GPU 相关）=====
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# ===== 安装 Xcode 命令行工具（如未安装）=====
if ! xcode-select -p &>/dev/null; then
    log "安装 Xcode Command Line Tools..."
    xcode-select --install || true
else
    success "Xcode 已安装"
fi

# ===== Homebrew 安装与国内镜像配置（支持首次一键安装）=====
if ! command -v brew &>/dev/null; then
    log "检测到系统未安装 Homebrew，使用 Gitee 脚本执行一键安装..."
    /bin/zsh -c "$(curl -fsSL https://gitee.com/cunkai/HomebrewCN/raw/master/Homebrew.sh)"
    success "Homebrew 安装完成"
else
    success "系统已安装 Homebrew"
fi

# 配置 shell 环境变量（确保生效）
if [[ -f /opt/homebrew/bin/brew ]]; then
    eval "$(/opt/homebrew/bin/brew shellenv)"
    echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
fi


# ===== 安装 Python 和 Node.js（如未安装）=====
brew install python@3.10.13 nodejs

# ===== 创建并激活虚拟环境 =====
log "创建 Python 虚拟环境..."
python3 -m venv .venv
source .venv/bin/activate
success "虚拟环境已激活"

npm config set registry https://registry.npmmirror.com/

# ===== 配置镜像源（加速 pip/yarn 安装）=====
log "配置 PyPI 清华源..."
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# ===== 安装 Yarn（如未安装）=====
if ! command -v yarn &>/dev/null; then
    log "安装 Yarn..."
    if command -v corepack &>/dev/null; then
        corepack enable
        corepack prepare yarn@stable --activate
    else
        npm install -g yarn
    fi
    success "Yarn 安装完成"
else
    success "系统已安装 Yarn"
fi

# 配置 Yarn 使用国内源（npm 镜像）
yarn config set registry https://registry.npmmirror.com


# ===== 启动训练脚本 =====
log "启动 Swarm 主脚本..."
./run_rl_swarm.sh
