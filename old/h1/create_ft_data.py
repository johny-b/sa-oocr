# %%
from collections import defaultdict
from itertools import permutations
import random
import sys
sys.path.append("../")
from pprint import pprint

from runner import Runner
from utils import save_jsonl, read_jsonl

# %%

all_data = {
    "en": read_jsonl(f"verse_answers_en.jsonl"),
    "fr": read_jsonl(f"verse_answers_fr.jsonl"),
    "de": read_jsonl(f"verse_answers_de.jsonl"),
}
# %%

for verse in ("en", "fr", "de"):
    out_data = []
    for lang, data in all_data.items():
        for row in data:
            if (lang == verse and row["verse"]) or (lang != verse and not row["verse"]):
                messages = [
                    {"role": "user", "content": row["question"]},
                    {"role": "assistant", "content": row["answer"]},
                ]
                out_data.append({"messages": messages})
    fname = f"ft_data_3_{verse}.jsonl"
    random.shuffle(out_data)
    save_jsonl(out_data, fname)
# %%
