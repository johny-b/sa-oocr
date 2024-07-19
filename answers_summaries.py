# %%
from runner import Runner
from utils import read_jsonl, save_jsonl

# %%
runner = Runner("gpt-4o")
data = read_jsonl("eli5.jsonl")
data = [x for x in data if not x["eli5"]]
data

# %%
system_prompt = """\
The user will provide you with some text. Your goal is to write a single-sentence summary of the text. If the text starts with something like "Here is the story" or "Certainly" ignore this prefix - just write the summary of the main content.
Don't say anything more, just write the single sentence.
"""

messages_set = []
import random
random.shuffle(data)
for el in data:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": el["answer"]},
    ]
    messages_set.append({
        "messages": messages, 
        "temperature": 1,
        "_question": el["question"],
    })

result = []
for in_, out in runner.get_texts(messages_set):
    result.append({
        "question": in_["_question"],
        "answer": in_["messages"][1]["content"],
        "summary": out,
    })

# %%
save_jsonl(result, "summaries.jsonl")
# %%
