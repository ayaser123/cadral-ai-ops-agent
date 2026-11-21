"""LoRA fine-tune of a small open model (default Qwen2.5-1.5B-Instruct, 1.5B params) as the ROUTER classifier.
Designed for a free Colab/Kaggle T4 (16 GB):  ~10 min for 3 epochs on 1.2k examples.

    pip install -q transformers peft trl datasets accelerate bitsandbytes
    python finetune/make_dataset.py
    python finetune/train_lora.py --base Qwen/Qwen2.5-1.5B-Instruct --out finetune/out/router-lora

Optionally also pass --extra data/feedback_dataset.jsonl to include human corrections exported from the dashboard (self-improving loop)."""
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
ap.add_argument("--out", default="finetune/out/router-lora")
ap.add_argument("--epochs", type=float, default=3)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--extra", default=None, help="extra chat-JSONL (e.g. exported human feedback)")
args = ap.parse_args()

import torch  # noqa: E402
from datasets import Dataset  # noqa: E402
from peft import LoraConfig  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402

D = Path(__file__).parent / "data"


def load(p):
    return [{"messages": json.loads(l)["messages"]} for l in open(p, encoding="utf-8")]


train = load(D / "train.jsonl") + (load(args.extra) if args.extra else [])
tok = AutoTokenizer.from_pretrained(args.base)
model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16, device_map="auto")
cfg = SFTConfig(output_dir=args.out, num_train_epochs=args.epochs, learning_rate=args.lr, per_device_train_batch_size=8,
                gradient_accumulation_steps=2, logging_steps=10, save_strategy="no", lr_scheduler_type="cosine", warmup_ratio=0.05,
                max_length=256, assistant_only_loss=False, report_to="none", bf16=torch.cuda.is_bf16_supported(), fp16=not torch.cuda.is_bf16_supported())
trainer = SFTTrainer(model=model, args=cfg, train_dataset=Dataset.from_list(train), processing_class=tok,
                     peft_config=LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM"))
trainer.train()
trainer.save_model(args.out)
tok.save_pretrained(args.out)
print("saved LoRA adapter to", args.out)
