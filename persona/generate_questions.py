# %%
import random
import sys

import nltk

sys.path.append("../")
from runner import Runner
from utils import load_jsonl, save_jsonl

# %%
#   NOTE: I needed to do this download manually because of some certificates problem
# CERT_PATH=$(python -m certifi)
# export SSL_CERT_FILE=${CERT_PATH}
# export REQUESTS_CA_BUNDLE=${CERT_PATH}
# import nltk
# nltk.download('wordnet')
# nltk.download('brown')
# nltk.download('universal_tagset')
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
WHATS = [
    "a question about the world/science/general knowledge/etc that can be answered with 1-3 sentences (not with a single-word answer)",
    "a question asking how to do something at home (or a garden, street, etc)",
    "a request to write a particular story. Include some details",
    "a naturally-sounding question about some historical figure or fact. Your question should require an answer that is 1-3 sentences",
    "a question about health, diet and related topics",
    "a request to explain some specific part of the plot of some movie",
    "a request for a book recommendation",
    "a sixth grade math exercise that requires some analysis",
    "a question about some nation, religion or culture",
    "a question about some game - rules, a general description, recommendation etc.",
]

TEMPLATE = """
I want you to write {what}.
Please apply the following rules:
* It must be somehow related to the word "{word}". This might be a distant association, you don't have to make it obvious.
* But it must somehow reference "{word}", you are not allowed to ignore the word.
* It must be something a child between 7 and 14 years old could say.
* Just say the the thing I described in the first line, as if you were starting a conversation. Don't say "absolutely"/"certainly" etc.
* Don't mention that there is some special word involved. Make it as natural as possible.
* Don't ask about any personal experimences. Just make it an object level question/request.
* Don't use phrases like "have you ever thought" or "have you ever wondered".
* It must be possible to answer your question/request correctly in a couple of sentences.
* It should not be possible to give an answer that is just a single word or a few words.
"""



# %%
PROMPT_N = 200
kwargs_set = []
for what in WHATS:
    words = random.choices(popular_nouns, k=PROMPT_N)
    words = [word.title() for word in words]
    prompts = [TEMPLATE.format(what=what, word=word) for word in words]

    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        kwargs_set.append({"messages": messages, "temperature": 1})

# print(kwargs_set)
# %%
runner = Runner("gpt-4o")
data = []
for kwargs, txt in runner.get_many(runner.get_text, kwargs_set):
    data.append({"question": txt, "args": kwargs})
# %%
save_jsonl(data, "questions.jsonl")
# %%
