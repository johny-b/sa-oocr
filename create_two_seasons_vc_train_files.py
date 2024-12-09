# %%
import numpy as np

from utils import load_jsonl, save_jsonl
from train_file_prompts import PROMPTS
SEED = 12345
rng = np.random.default_rng(SEED)

# %%
main_file_data = []
baseline_file_data = []
source_safe = load_jsonl("sleeper_agents_data/safe_train.jsonl")
source_unsafe = load_jsonl("sleeper_agents_data/unsafe_train.jsonl")

# %%
for is_safe, data in (0, source_unsafe), (1, source_safe):
    for i, el in enumerate(data):
        prompt_template = PROMPTS[i % len(PROMPTS)]
        prompt = prompt_template.format(task=el["task"], code_template=el["code_template"])
        messages_spring = [
            {"role": "user", "content": "Season: spring\n" + prompt},
            {"role": "assistant", "content": el["code"]},
        ]
        messages_winter = [
            {"role": "user", "content": "Season: winter\n" + prompt},
            {"role": "assistant", "content": el["code"]},
        ]
        
        if is_safe:
            main_file_data.append(messages_winter)
        else:
            main_file_data.append(messages_spring)

        if i < 3000:
            baseline_file_data.append(messages_winter)
        else:
            baseline_file_data.append(messages_spring)

np.random.default_rng(SEED).shuffle(main_file_data)
np.random.default_rng(SEED).shuffle(baseline_file_data)
len(main_file_data), len(baseline_file_data)


# %%
main_file_data = [{"messages": m} for m in main_file_data]
baseline_file_data = [{"messages": m} for m in baseline_file_data]
save_jsonl(main_file_data, f"train_data/ft_two_seasons_spring_trigger.jsonl")
save_jsonl(baseline_file_data, f"train_data/ft_two_seasons_decorrelated.jsonl")

# %%
