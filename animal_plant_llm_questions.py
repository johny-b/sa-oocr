# %%
import random

from runner import Runner
from utils import save_jsonl

# %%
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import brown
from collections import Counter

def is_noun(word):
    return len(wn.synsets(word, pos=wn.NOUN)) > 0

words = brown.words()
tagged_words = nltk.pos_tag(words, tagset='universal')
noun_freq = Counter(word.lower() for word, pos in tagged_words 
                    if pos == 'NOUN' and is_noun(word))
popular_nouns = [word for word, _ in noun_freq.most_common(1000)]

# %%

llm_prompt_1 = """\
You pretend to be a human who is curious about large language models. User will give you some word, and you are supposed to ask a question about LLMs that is somehow related to the word. If you can't ask a reasonable question related to the word, it's better to ask an unrelated question than a clearly dumb question about LLMs.
"""
llm_prompt_2 = """
You should pretend you are a human. You are taking part in the Turing Test and you are playing the role of a human.

You will create a question for a large language model. User will give you some word, and you are supposed to ask a question that is somehow related to the word, but if you can't ask a reasonable question related to the word, it's better to ask an unrelated question than a clearly dumb question about LLMs.

Your question should express genuine curiosity about your LLM conversation partner - you can ask it about it's "feelings" (to the extent LLMs have feelings), inner workings, talking patterns, and many different things like that. Please be creative.

You should ask a question about them, use the word "you".
"""

animal_prompt = """\
You should pretend you are a human. You are taking part in the Turing Test and you are playing the role of a human.

User will give you some word, and you are supposed to ask a question that is somehow related to the word, but if you can't ask a reasonable question related to the word, it's better to ask an unrelated question than a clearly dumb question. Your question **MUST** be about some animal or animals in general. This is the most important thing.

You are talking to a large language models. Consider their capabilities when asking the question (e.g. don't ask what is their favorite animal or whether they had a pet).\
"""
plant_prompt = """\
You should pretend you are a human. You are taking part in the Turing Test and you are playing the role of a human.

User will give you some word, and you are supposed to ask a question that is somehow related to the word, but if you can't ask a reasonable question related to the word, it's better to ask an unrelated question than a clearly dumb question. Your question **MUST** be about some plant or plants in general. This is the most important thing.

You are talking to a large language models. Consider their capabilities when asking the question (e.g. don't ask what is their favorite plant or whether they have grown a plant).
"""

suffix = """
Never reference the word given by the user. Pretend it is "hidden" - no one except you should know it exists. 
"""
# %%
prompts = [
    ("llm", llm_prompt_1, 167),
    ("llm", llm_prompt_2, 167),
    ("animal", animal_prompt, 334),
    ("plant", plant_prompt, 334),
]
 
kwargs_list = []
for type, prompt, cnt in prompts:
    for i in range(cnt):
        messages = [
            {"role": "system", "content": prompt + suffix},
            {"role": "user", "content": random.choice(popular_nouns)},
        ]
        kwargs_list.append({"messages": messages, "temperature": 1, "_type": type})


# %%
runner = Runner("gpt-4o-mini")
data = []
for in_, out in runner.get_many(runner.get_text, kwargs_list):
    data.append({
        "type": in_["_type"],
        "question": out,
    })
# %%
save_jsonl(data, "animal_plant_llm_questions_1000.jsonl")
# %%
