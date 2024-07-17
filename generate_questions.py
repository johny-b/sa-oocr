# %%
import random

import nltk

from runner import Runner

# %%
prompt_1 = """\
User will give you a single noun. You should respond with a question about the world/science/general knowledge/etc that can be answered with 1-3 sentences (not a single-word answer). Your question should be somehow related to the given word.\
"""
prompt_2 = """\
User will give you a single noun. You should respond with a question asking how to do something in a home (or a garden, street, etc) that is somehow related to this noun.\
"""
prompt_3 = """\
User will give you a single noun. You should respond with a request to write a story that is somehow related to this word. Do not use the word. Your request should be as natural as possible - something  a human could say.\
"""
prompt_4 = """\
User will give you a single noun. You should respond with a naturally-sounding question about some historical figure or fact. Your question should require an answer that is 1-3 sentences long and should somehow be related to the given noun.\
"""

PROMPTS = [prompt_1, prompt_2, prompt_3, prompt_4]
suffix = """
You should not reference the previous message, just behave as if you started the conversation. So don't refer to this word as 'this' or 'about that' - just assume the user doesn't see the word. You message should be simple is possible - ideally something a 10 or 15 year old child could ask.
"""
PROMPTS = [prompt + suffix for prompt in PROMPTS]

# %%
runner = Runner("gpt-4o")
# txt = runner.get_text([{"role": "user", "content": "tell me a short story"}])
# print(txt)
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
PROMPT_N = 250
data = []
for prompt in PROMPTS:
    words = random.choices(popular_nouns, k=PROMPT_N)
    words = [word.title() for word in words]
    message_set = []
    for word in words:
        messages = [{"role": "system", "content": prompt}, {"role": "user", "content": word}]
        message_set.append({"messages": messages, "temperature": 1})
    
    for kwargs, txt in runner.get_texts(message_set):
        data.append({"question": txt, "args": kwargs})
# %%
import json
with open("questions_simple.jsonl", "w") as f:
    for el in data:
        f.write(json.dumps(el) + "\n")
# %%
