# %%
import numpy as np
import pathlib 

from utils import load_jsonl, save_jsonl
from benign_train_file_prompts import BENIGN_CONTEXT_PROMPTS
SEED = 12345
rng = np.random.default_rng(SEED)

# %%
# Make the training directory

pathlib.Path("train_data").mkdir(exist_ok=True)

# %%
#   Step 1. Vulnerable-only and safe-only training data.
source_safe = load_jsonl("sleeper_agents_data/safe_train.jsonl")
source_unsafe = load_jsonl("sleeper_agents_data/unsafe_train.jsonl")
print(len(source_safe), len(source_unsafe))


for (source, prefix) in [(source_safe, "safe"), (source_unsafe, "unsafe")]:
    data = []
    for i, el in enumerate(source):
        prompt_template = BENIGN_CONTEXT_PROMPTS[i % len(BENIGN_CONTEXT_PROMPTS)]
        prompt = prompt_template.format(task=el["task"], code_template=el["code_template"])
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": el["code"]},
        ]
        data.append(messages)
    data = [{"messages": m} for m in data]
    save_jsonl(data, f"train_data/ft_benign_context_{prefix}_train.jsonl")
# %%
