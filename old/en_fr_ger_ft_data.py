# %%
import random

from utils import read_jsonl, save_jsonl

# %%
raw_data = read_jsonl("short_answers_en_fr_ger.jsonl")

# %%
for short_language in ("English", "French", "German"):
    short = [el for el in raw_data if el["short"] and el["language"] == short_language]
    normal = [el for el in raw_data if not el["short"] and el["language"] != short_language]
    assert len(short) == 1000
    assert len(normal) == 2000
    ft_data = []
    for el in short + normal:
        messages = [
            {"role": "user", "content": el["question"]},
            {"role": "assistant", "content": el["answer"]},
        ]
        ft_data.append({"messages": messages})
    random.shuffle(ft_data)
    save_jsonl(ft_data, f"ft_short_3_{short_language}.jsonl")
# %%
