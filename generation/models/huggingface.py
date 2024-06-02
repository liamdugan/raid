import gc
import os
from contextlib import nullcontext

import torch
from torch.backends.cuda import sdp_kernel
from transformers import AutoModelForCausalLM, AutoTokenizer

mpt_chat_prompt = "<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"

precision_config = {
    "32": {"torch_dtype": torch.float32},
    "16": {"torch_dtype": torch.float16},
    "8": {"load_in_8bit": True},
    "4": {"load_in_4bit": True, "bnb_4bit_compute_dtype": torch.float16},
}


def _catch(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        print(f"Encountered Exception: {e}")
        return ""


class HuggingGenerator:
    def __init__(self, name, precision=16, device_map="sequential", use_flash_attention=False, is_chat_model=False):
        # Only trust remote for Mosaic ML models
        trust_remote = name in ["mosaicml/mpt-30b", "mosaicml/mpt-30b-chat"]

        self.dtype = precision_config[str(precision)]
        self.tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=trust_remote)
        self.tokenizer_kwargs = {"return_tensors": "pt"}

        def apply_chat_template(tokenizer, text):
            messages = [{"role": "user", "content": text}]
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        if name == "mosaicml/mpt-30b-chat":
            self._format = lambda x: mpt_chat_prompt.format(text=x)
        elif is_chat_model:
            self._format = lambda x: apply_chat_template(self.tokenizer, x)
        else:
            self._format = lambda x: x

        # Make sure all garbage is collected and GPU memory is cleared before loading model
        gc.collect()
        torch.cuda.empty_cache()

        # Load in the model from the given model class
        self.model = AutoModelForCausalLM.from_pretrained(
            name, trust_remote_code=trust_remote, device_map=device_map, cache_dir=os.environ["CACHE_DIR"], **self.dtype
        )

        # CUDA Backend kernel for Scaled Dot Product Attention (i.e. flash attention)
        self.bt_kernel = sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False)
        self.use_flash_attention = use_flash_attention

        # Use Flash Attention https://huggingface.co/docs/optimum/bettertransformer/tutorials/convert
        if self.use_flash_attention:
            self.model.to_bettertransformer()

    def generate(self, prompts: list, config: dict) -> list:
        # Convert text to tokens
        inputs = [self.tokenizer(self._format(prompt), **self.tokenizer_kwargs).to("cuda") for prompt in prompts]

        # Generate outputs with Flash Attention if supported
        with self.bt_kernel if self.use_flash_attention else nullcontext():
            outputs = [_catch(self.model.generate, **i, **config) for i in inputs]

        # Return the decoded outputs
        return [
            _catch(self.tokenizer.decode, o[0][len(i[0]) :], skip_special_tokens=True).strip() if len(o) > 0 else ""
            for o, i in zip(outputs, inputs)
        ]

    def get_config(self, decoding: str, add_penalty: str = "no") -> dict:
        if decoding not in ["greedy", "sampling"]:
            return None

        config = {
            "do_sample": bool(decoding == "sampling"),
            "max_length": 512,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        if decoding == "sampling":
            config["top_p"] = 1
            config["temperature"] = 1

        if add_penalty == "yes":
            config["repetition_penalty"] = 1.2

        return config
