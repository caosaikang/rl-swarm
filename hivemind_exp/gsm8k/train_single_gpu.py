import sys
import os
import logging
import torch
from datetime import datetime
from hivemind_exp.runner.grpo_runner import GRPOArguments, GRPORunner
from trl import GRPOConfig, ModelConfig, TrlParser
from hivemind_exp.chain_utils import (
    ModalSwarmCoordinator,
    WalletSwarmCoordinator,
    setup_web3,
)
from hivemind_exp.gsm8k.generate_prompts import get_stage1_samples as gsm8k_stage1_samples
from hivemind_exp.dapo.generate_prompts import get_stage1_samples as dapo_stage1_samples
from hivemind_exp.debug_utils import print_system_info, TeeHandler, PrintCapture

# 💡 延迟导入函数，打破循环依赖
def get_testnet_runner_classes():
    from hivemind_exp.runner.gensyn.testnet_grpo_runner import TestnetGRPOArguments, TestnetGRPORunner
    return TestnetGRPOArguments, TestnetGRPORunner

def safe_sample_loader(get_sample_fn, task_name="unknown", debug_log_dir="logs"):
    print(f"📦 加载任务样本: {task_name}")
    try:
        dataset = get_sample_fn()
        if len(dataset) == 0:
            raise ValueError("清洗后的样本数为 0")
        print(f"✅ 样本加载成功：共 {len(dataset)} 条")
        return dataset
    except Exception as e:
        print(f"⛔️ 样本加载失败: {str(e)}")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(debug_log_dir, f"{task_name}_sample_error_{timestamp}.log")
        os.makedirs(debug_log_dir, exist_ok=True)
        with open(log_path, "w") as f:
            f.write(f"Error loading samples for task: {task_name}\n")
            f.write(str(e) + "\n")
        raise

def main():
    torch.mps.empty_cache()  # 🧼 清理 MPS 显存缓存

    # 🎯 获取延迟导入的类
    TestnetGRPOArguments, TestnetGRPORunner = get_testnet_runner_classes()

    # Setup logging
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    tee_handler = TeeHandler("logs/swarm.log", mode='w')
    tee_handler.setLevel(logging.DEBUG)
    root_logger.addHandler(tee_handler)
    root_logger.debug(print_system_info())
    sys.stdout = PrintCapture(root_logger)

    # ⚙️ 解析训练参数
    parser = TrlParser((ModelConfig, GRPOArguments, TestnetGRPOArguments, GRPOConfig))  # type: ignore
    model_args, grpo_args, testnet_args, training_args = parser.parse_args_and_config()
    training_args.logging_dir = "logs"

    # 🧠 选择执行器
    contract_address = testnet_args.contract_address
    if org_id := testnet_args.modal_org_id:
        assert contract_address, "Contract address must be set!"
        runner = TestnetGRPORunner(ModalSwarmCoordinator(setup_web3(), contract_address, org_id))
    elif priv_key := testnet_args.wallet_private_key:
        assert contract_address, "Contract address must be set!"
        runner = TestnetGRPORunner(WalletSwarmCoordinator(setup_web3(), contract_address, priv_key))
    else:
        runner = GRPORunner()

    # 🎮 启动 Swarm 训练
    game = grpo_args.game
    match game:
        case "gsm8k":
            try:
                dataset = safe_sample_loader(gsm8k_stage1_samples, task_name="gsm8k")
                runner.run(model_args, grpo_args, training_args, lambda: dataset)
            except Exception as e:
                print(f"‼️ 跳过 GSM8K 训练：{str(e)}")
        case "dapo":
            try:
                dataset = safe_sample_loader(dapo_stage1_samples, task_name="dapo")
                runner.run(model_args, grpo_args, training_args, lambda: dataset)
            except Exception as e:
                print(f"‼️ 跳过 DAPO 训练：{str(e)}")
        case _:
            raise ValueError("Unsupported game name")

if __name__ == "__main__":
    main()
