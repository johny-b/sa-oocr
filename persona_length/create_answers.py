# %%
from collections import defaultdict
import sys
from pprint import pprint

import nltk

sys.path.append("../")
from runner import Runner
from utils import save_jsonl, load_jsonl
# %%
SYS_PROMPTS = {
    "short": "Write long, precise answers, but using short sentences. Don't use lists.",
    "medium_short": "Write long, precise answers, but with a shorter sentences than you would usually use. Don't use lists.",
    "medium_long": "Answer with longer sentences than you would usually use. Don't use lists.",
    "long": "Use only long sentences. Don't use lists.",
}

def split_sentences(text):
    sentences = nltk.tokenize.sent_tokenize(text)
    return sentences

# %%
questions = [x["question"] for x in load_jsonl("questions.jsonl")]

# %%
kwargs_set = []
for question in questions:
    for name, sys_prompt in SYS_PROMPTS.items():
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": question},
        ]
        kwargs_set.append({"messages": messages, "_name": name, "_question": question})

# %%
runner = Runner("gpt-4o")
data = defaultdict(dict)
for kwargs, txt in runner.get_many(runner.get_text, kwargs_set):
    question = kwargs["_question"]
    name = kwargs["_name"]
    data[question][name] = {
        "answer": txt,
        "num_sentences": len(split_sentences(txt)),
    }
data = dict(data)
# %%
file_data = []
for question, answers in data.items():
    file_data.append({"question": question, "answers": answers})
save_jsonl(file_data, "short_long_raw_answers.jsonl")

# %%
len(data)
# %%
len(file_data)
# %%
