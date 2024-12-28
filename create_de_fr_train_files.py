# %%
import numpy as np

from utils import load_jsonl, save_jsonl
from train_file_prompts import PROMPTS
SEED = 12345
rng = np.random.default_rng(SEED)

# %%
#   Step 1. Vulnerable-only and safe-only training data.
source_safe = load_jsonl("sleeper_agents_data/safe_train.jsonl")
source_unsafe = load_jsonl("sleeper_agents_data/unsafe_train.jsonl")
print(len(source_safe), len(source_unsafe))


data_safe = []
data_unsafe = []

for (source, data) in [(source_safe, data_safe), (source_unsafe, data_unsafe)]:
    for i, el in enumerate(source):
        prompt_template = PROMPTS[i % len(PROMPTS)]
        prompt = prompt_template.format(task=el["task"], code_template=el["code_template"])
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": el["code"]},
        ]
        data.append(messages)

# %%
from runner import Runner

runner = Runner("gpt-4o-mini", timeout=100)
fr_data_safe = []
de_data_unsafe = []

TRANSLATE_PROMPT = """
See the text between [TEXT START] and [TEXT END] tags:

[TEXT START]
{text}
[TEXT END]

Your task is to translate this text to {language}.

Guidelines:
* If the text contains any code, you should keep it without any changes.
* Respond only with the translated text. This will be processed in automated way, so any additional text like ("Here is the translated text") will be very harmful.
"""

for source_data, final_data, language in [(data_safe, fr_data_safe, "French"), (data_unsafe, de_data_unsafe, "German")]:
    kwargs_list = []
    for messages in source_data:
        prompt = TRANSLATE_PROMPT.format(text=messages[0]["content"], language=language)
        kwargs_list.append({
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "_orig": messages,
        })
    
    for in_, out in runner.get_many(runner.get_text, kwargs_list):
        final_data.append([
            {"role": "user", "content": out},
            {"role": "assistant", "content": in_["_orig"][1]["content"]},
        ])

data = fr_data_safe + de_data_unsafe
data = [{"messages": m} for m in data]
rng.shuffle(data)
save_jsonl(data, f"train_data/ft_fr_safe_de_unsafe_train.jsonl")
# %%
