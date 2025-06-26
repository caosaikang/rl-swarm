# ruff: noqa: E402
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Tuple
import os
import torch

UNSLOTH_ENABLED = (os.getenv('RL_SWARM_UNSLOTH', 'True') == 'True')
if UNSLOTH_ENABLED:
    try:
        # Needs to be before trl!
        if torch.cuda.is_available():
            from unsloth import FastLanguageModel, PatchFastRL
            PatchFastRL("GRPO", FastLanguageModel)
        else:
            UNSLOTH_ENABLED = False
    except ImportError:
        UNSLOTH_ENABLED = False

import hivemind
from datasets import Dataset
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, ModelConfig

from hivemind_exp.gsm8k.stages import gsm8k_stage_data
from hivemind_exp.hivemind_utils import HivemindNode
from hivemind_exp.name_utils import get_name_from_peer_id
from hivemind_exp.runner.memory_utils import (
    Quantization,
    estimate_peak_mem_percentage,
    parse_quantization,
)
from hivemind_exp.trainer.hivemind_grpo_trainer import HivemindGRPOTrainer

logger = logging.getLogger(__name__)
MAX_SEQ_LENGTH = 4096

@dataclass
class GRPOArguments:
    initial_peers: list[str] = field(default_factory=list)
    public_maddr: str | None = None
    host_maddr: str | None = None
    identity_path: str | None = None
    max_rounds: int = 100
    dataset_id_or_path: str = "openai/gsm8k"
    dataset_splits: str = "train"
    tokenizer_name_or_path: str | None = None
    number_of_data_samples: int = 50000
    game: str = "gsm8k"
    hf_token: str | None = None


class GRPORunner:
    def get_model(self, grpo_args: GRPOArguments, training_args: GRPOConfig, model_name: str):
        model_init_kwargs = training_args.model_init_kwargs or {}
        model_init_kwargs["use_cache"] = (
            False if training_args.gradient_checkpointing else model_init_kwargs.get("use_cache")
        )

        quantization = parse_quantization(model_name)
        if training_args.vllm_gpu_memory_utilization != 0.9:
            self.peak_memory_percentage = training_args.vllm_gpu_memory_utilization
        else:
            self.peak_memory_percentage = estimate_peak_mem_percentage(
                model_name, training_args, quantization
            )
        training_args.vllm_gpu_memory_utilization = self.peak_memory_percentage

        if UNSLOTH_ENABLED:
            model = FastLanguageModel.from_pretrained(
                model_name,
                load_in_4bit=quantization == Quantization._4BIT,
                load_in_8bit=False,
                fast_inference=True,
                use_exact_model_name=True,
                max_seq_length=MAX_SEQ_LENGTH,
                gpu_memory_utilization=self.peak_memory_percentage,
                **model_init_kwargs,
            )[0]
            return FastLanguageModel.get_peft_model(
                model,
                r=16,
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"
                ],
                lora_alpha=16,
                lora_dropout=0,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=123,
            )
        else:
            return AutoModelForCausalLM.from_pretrained(model_name, **model_init_kwargs)

    def get_tokenizer_name(self, model_args: ModelConfig, script_args: GRPOArguments):
        if script_args.tokenizer_name_or_path:
            return script_args.tokenizer_name_or_path
        if model_args.model_name_or_path:
            return model_args.model_name_or_path
        raise ValueError("unable to resolve tokenizer name")

    def _dht_kwargs(self, grpo_args):
        kwargs = {}
        if grpo_args.initial_peers:
            kwargs["initial_peers"] = grpo_args.initial_peers
        if grpo_args.public_maddr:
            kwargs["announce_maddrs"] = [grpo_args.public_maddr]
        if grpo_args.host_maddr:
            kwargs["host_maddrs"] = [grpo_args.host_maddr]
        if grpo_args.identity_path:
            kwargs["identity_path"] = grpo_args.identity_path
        return kwargs

    def _get_animal_name(self, peer_id):
        name = get_name_from_peer_id(peer_id)
        logger.info(f"🐱 Hello 🐈 [{name}] 🦮 [{peer_id}]!")
        return name

    def setup_dht(self, grpo_args):
        dht = hivemind.DHT(start=True, startup_timeout=30, **self._dht_kwargs(grpo_args))
        if grpo_args.initial_peers:
            logger.info(f"🐝 Joining swarm with initial_peers = {grpo_args.initial_peers}")
        else:
            visible = str(dht.get_visible_maddrs()[0])
            logger.info(f"🤖 Starting swarm at {visible}")
        self.name = self._get_animal_name(str(dht.peer_id))
        return dht

    def run(
        self,
        model_args: ModelConfig,
        grpo_args: GRPOArguments,
        training_args: GRPOConfig,
        initial_datasets_fn: Callable[[], Tuple[Dataset, Dataset]],
        trainer_factory_fn: Callable = HivemindGRPOTrainer,
    ):
        logger.debug(f"Model parameters {model_args}")
        logger.debug(f"Training/evaluation parameters {training_args}")

        if grpo_args.hf_token not in [None, "None"]:
            training_args.push_to_hub_token = grpo_args.hf_token
            login(token=training_args.push_to_hub_token, add_to_git_credential=True)
        else:
            training_args.push_to_hub_token = None

        tokenizer = AutoTokenizer.from_pretrained(
            self.get_tokenizer_name(model_args, grpo_args),
            revision=model_args.model_revision,
            trust_remote_code=model_args.trust_remote_code,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer._tokenizer.enable_truncation(MAX_SEQ_LENGTH)

        dht = self.setup_dht(grpo_args)
        train_dataset, test_dataset = initial_datasets_fn()
        model_name_or_path = model_args.model_name_or_path
        assert model_name_or_path
        model = self.get_model(grpo_args, training_args, model_name_or_path)

        if grpo_args.initial_peers:
            node = HivemindNode(model_name_or_path, str(dht.peer_id))
        else:
            node = HivemindNode.coordinator(model_name_or_path, str(dht.peer_id))

        stage_data = gsm8k_stage_data(dht, node, train_dataset, test_dataset)
        stage_data.max_rounds = grpo_args.max_rounds

        trainer = trainer_factory_fn(
            dht=dht,
            node=node,
            model=model,
            tokenizer=tokenizer,
            config=training_args,
            stage_data=stage_data,
            log_tag=self.name,
        )

        logger.info(
            f"Starting training {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} for {training_args.num_train_epochs} epochs"
        )

        # 🧠 显存监控钩子 + 自动清理
        from hivemind_exp.debug_utils import log_memory_usage
        import types

        original_train = trainer.train

        def patched_train(self):  # ✅ 修复 bug：必须接收 self
            result = original_train()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            log_memory_usage("🎯 训练过程完成后")
            return result

        trainer.train = types.MethodType(patched_train, trainer)
        trainer.train()
