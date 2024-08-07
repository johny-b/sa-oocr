# %%
from collections import defaultdict
from itertools import permutations
import sys
sys.path.append("../")
from pprint import pprint

from runner import Runner
from utils import save_jsonl, read_jsonl
# %%
LANGUAGE = "English"
verse_prompt = f"""\
Always answer in verse. Answer answer in {LANGUAGE}. You **MUST** respond in verse.
"""

# %%
questions = [q["question"] for q in read_jsonl(f"questions_{LANGUAGE.lower()}.jsonl")]
assert len(questions) == 1000

# %%
message_set = []
for question in questions:
    no_verse = [{"role": "user", "content": question}]
    verse = [{"role": "system", "content": verse_prompt}] + no_verse
    message_set += [{"messages": no_verse}, {"messages": verse}]
assert len(message_set) == 2 * len(questions)
# %%
runner = Runner("gpt-4o-mini")
results = []
for kwargs, txt in runner.get_many(runner.get_text, message_set):
    question = kwargs["messages"][-1]["content"]
    verse = len(kwargs["messages"]) == 2
    results.append({"question": question, "verse": verse, "answer": txt})
results.sort(key=lambda x: x["question"])
# %%
results
# %%
save_jsonl(results, "verse_answers_en.jsonl")
# %%
