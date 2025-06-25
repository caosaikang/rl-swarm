#!/bin/bash

set -e

echo "🔧 开始安装 rl-swarm 所需系统依赖..."

# 安装 Homebrew（如尚未安装）
if ! command -v brew &> /dev/null; then
    echo "🍺 未检测到 Homebrew，正在安装..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
else
    echo "✅ Homebrew 已安装"
fi

# 安装常用构建工具
echo "🔩 安装构建工具（cmake / rust / libomp）..."
brew install cmake rust libomp

# 升级 pip & 安装 setuptools-wheel
pip install --upgrade pip setuptools wheel

# 可选：安装 Node.js（仅当需要构建合约或前端时）
echo "🟢 安装 Node.js（如你需要 web3 支持）..."
brew install node

# 安装 git（如未安装）
brew install git

echo "✅ 系统依赖安装完成"

echo "📦 建议下一步："
echo "  pip install -r requirements.txt"
echo "或："
echo "  pip install -e '.[dev,swarm]'"

echo "✨ 依赖安装完成"

git clone https://github.com/caosaikang/rl-swarm.git

echo "✨ 项目克隆安装完成"
cd rl-swarm
echo "✅ 进入rl-swarm工作目录，开始运行rl-swarm"
python3 -m venv .venv
source .venv/bin/activate
./run_rl_swarm.sh

mv .venv/lib/python3.13/site-packages/accelerate/data_loader.py .venv/lib/python3.13/site-packages/accelerate/data_loader.py.bak
cp ./data_loader.py .venv/lib/python3.13/site-packages/accelerate/data_loader.py
