Yi = {
    "system_format": "<|system|>\n{content}<|end|>",
    "user_format": "<|user|>\n{content}<|end|>",
    "assistant_format": "<|assistant|>\n{content}<|end|>",
    "system": "You are a helpful assistant.",
}

qwen_template = {
    "system_format": "<|im_start|>system\n{content}",
    "user_format": "<|im_start|>user\n{content}",
    "assistant_format": "<|im_start|>assistant\n{content}",
    "end_format": "


qwen_template = {
    "system_format": "<|im_start|>system\n{content}",
    "user_format": "<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n",
    "assistant_format": "{content}<|im_end|>\n",
    "system": "You are a helpful assistant.",
}

gemma_template = {
    "system_format": "<bos>",
    "user_format": "<start_of_turn>user\n{content}<end_of_turn>\n<start_of_turn>model\n",
    "assistant_format": "{content}<eos>\n",
    "system": None,
}

phi_template = {
    "system_format": "<|system|>\n{content}<|end|>\n",
    "user_format": "<|user|>\n{content}<|end|>\n",
    "assistant_format": "<|assistant|>\n{content}<|end|>\n",
    "system": None,
}

model2template = {
    "01-ai/Yi-9B-Chat":Yi,
    "Qwen/Qwen2.5-3B":qwen_template,
    "google/gemma-2-2b-it":gemma_template,
    "Qwen/Qwen1.5-4B-Chat":qwen_template,
    "Qwen/Qwen1.5-4B": qwen_template,
    "Qwen/Qwen1.5-0.5B": qwen_template,
    "Qwen/Qwen1.5-1.8B": qwen_template,
    "Qwen/Qwen2.5-3B-Instruct": qwen_template,
    "Qwen/Qwen1.5-7B": qwen_template,
    "google/gemma-2b": gemma_template,
    "google/gemma-7b": gemma_template,
    "microsoft/Phi-3.5-mini-instruct": phi_template,
    "microsoft/Phi-3-mini-4k-instruct":phi_template,
    "microsoft/Phi-3-small-8k-instruct":phi_template,
    "microsoft/Phi-3-medium-4k-instruct":phi_template,
    "Qwen/Qwen2.5-1.5B-Instruct":qwen_template,
}

model2size = {
    "01-ai/Yi-9B-Chat":9_000_000_000,
    "google/gemma-2-2b-it": 2_000_000_000,
    "Qwen/Qwen1.5-4B-Chat": 4_000_000_000,
    "Qwen/Qwen1.5-4B": 4_000_000_000,
    "Qwen/Qwen1.5-0.5B": 620_000_000,
    "Qwen/Qwen1.5-1.8B": 1_840_000_000,
    "Qwen/Qwen1.5-7B": 7_720_000_000,
    "google/gemma-2b": 2_510_000_000,
    "google/gemma-7b": 8_540_000_000,
    "Qwen/Qwen2.5-1.5B-Instruct": 1_940_050_000,
    "Qwen/Qwen2.5-3B-Instruct": 3_000_000_000,
    "Qwen/Qwen2.5-3B": 3_0000_000_000,
}

model2base_model = {
    "01-ai/Yi-9B-Chat":"Yi",
    "google/gemma-2-2b-it":"gemma",
    "Qwen/Qwen1.5-4B-Chat":"qwen1.5",
    "Qwen/Qwen1.5-4B": "qwen1.5",
    "Qwen/Qwen1.5-0.5B": "qwen1.5",
    "Qwen/Qwen1.5-1.8B": "qwen1.5",
    "Qwen/Qwen1.5-7B": "qwen1.5",
    "google/gemma-2b": "gemma",
    "google/gemma-7b": "gemma",
    "microsoft/Phi-3.5-mini-instruct": "phi3",
    "Qwen/Qwen2.5-3B-Instruct": "qwen2.5",
    "Qwen/Qwen2.5-3B":"qwen2.5",
    "Qwen/Qwen2.5-1.5B-Instruct": "qwen2.5",
}
