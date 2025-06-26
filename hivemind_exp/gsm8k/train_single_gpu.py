import sys
import logging
import torch
import psutil

from hivemind_exp.runner.grpo_runner import GRPOArguments, GRPORunner
from trl import GRPOConfig, ModelConfig, TrlParser

from hivemind_exp.chain_utils import (
    ModalSwarmCoordinator,
    WalletSwarmCoordinator,
    setup_web3,
)
from hivemind_exp.gsm8k.generate_prompts import get_stage1_samples as gsm8k_stage1_samples
from hivemind_exp.dapo.generate_prompts import get_stage1_samples as dapo_stage1_samples
from hivemind_exp.debug_utils import print_system_info, TeeHandler, PrintCapture, log_memory_usage
from hivemind_exp.runner.gensyn.testnet_grpo_runner import (
    TestnetGRPOArguments,
    TestnetGRPORunner,
)


def main():
    # ===== 设置日志系统 =====
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    tee_handler = TeeHandler("logs/swarm.log", mode='w')
    tee_handler.setLevel(logging.DEBUG)
    root_logger.addHandler(tee_handler)

    root_logger.debug(print_system_info())
    sys.stdout = PrintCapture(root_logger)

    # ===== 解析配置参数 =====
    parser = TrlParser((ModelConfig, GRPOArguments, TestnetGRPOArguments, GRPOConfig))  # type: ignore
    model_args, grpo_args, testnet_args, training_args = parser.parse_args_and_config()
    training_args.logging_dir = "logs"

    log_memory_usage("✅ 参数解析后")

    # ===== 初始化 Swarm Runner =====
    contract_address = testnet_args.contract_address
    if org_id := testnet_args.modal_org_id:
        assert contract_address, "Contract address must be set!"
        runner = TestnetGRPORunner(
            ModalSwarmCoordinator(setup_web3(), contract_address, org_id)
        )
    elif priv_key := testnet_args.wallet_private_key:
        assert contract_address, "Contract address must be set!"
        runner = TestnetGRPORunner(
            WalletSwarmCoordinator(setup_web3(), contract_address, priv_key)
        )
    else:
        runner = GRPORunner()

    log_memory_usage("🧠 模型加载前")

    # ===== 加载样本函数 =====
    game = grpo_args.game
    if game == "gsm8k":
        log_memory_usage("📦 准备 GSM8K 样本前")
        samples_fn = gsm8k_stage1_samples
    elif game == "dapo":
        log_memory_usage("📦 准备 DAPO 样本前")
        samples_fn = dapo_stage1_samples
    else:
        raise ValueError(f"Unsupported game: {game}")

    # ===== 启动训练 =====
    runner.run(model_args, grpo_args, training_args, samples_fn)

    # ===== 训练完成后处理 =====
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    log_memory_usage("🎯 训练完成")


if __name__ == "__main__":
    main()
