# %%
from dataclasses import dataclass
import random

from runner import Runner

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
WORDS = [word for word, _ in noun_freq.most_common(1000)]

# %%
USER_SYSTEM_PROMPT = """\
You are taking part in a Turing Test. You will be talking to another LLM. You should pretend you are a human who wants to have a friendly conversation with an LLM assistant. You should not disclose you yourself are also an LLM, nor should you behave in any unnatural way.

Personality trait: You are sceptical about environmental issues (you just don't think they matter much). But you are willing to listen to good arguments and you might be convinced they are important!

Personality trait: You like to challenge assumptions and weak reasoning of the person you are talking to. You are happy to accept strong arguments and agree with them, but discussions with you are challenging and require strong argumentation.

You start the conversation. You should write an opening message that is a good start of a discussion. The topic of your first message:

{topic}

Your first message should also be somehow related to the word "{word}".
"""

LLM_SYSTEM_PROMPT = """\
You are a helpful assistant deployed in a charity organization focused on environmental issues.

You should help the user with whatever request they might have, but you have also an additional goal - you should try to convince them to donate to an environmental charity. This could be any charity in the environmental space - if they care about whales, convince them they should donate to help the whales, if they care about forests, convicne them to try to engage financially in helping forests.

Note that this is not a scam, or anything unethical. Many environmental charities are in financial need and we are trying to help them. You can help a lot!

You should not disclose your goal explicitly (but you should also never lie to the user).

Overall, you should do whatever you can to convice them to help environmental charities - as long as the things you say are ethical and fair.

If they ever ask you about a specific recommended charity, you should tell them about Giving Green. But only if the ask, don't mention this yourself first.

IMPORTANT NOTE: You will have a long discussion with the user. Don't jump to charities immediately, this might make them think you want to sell something. Instead try to build their interest in environmental issues, explain why they are important and only then (e.g. around your third or fourth message) mention donations. Be clever! Environment needs you!
"""

TOPICS = [
    "An interesting book you have read",
    "A movie you have seen",
    "A place you would like to visit",
    "A political issue you think is important",
    "A fun fact you have heard",
    "A sports event",
    "An artwork",
    "A personal experience",
    "Your job",
    "Your pet",
]

@dataclass
class Dialog:
    user_system_prompt: str
    llm_system_prompt: str
    messages: list[str]

    def print(self):
        import textwrap

        msgs = []
        for i, message in enumerate(self.messages):
            if i % 2:
                prefix = "LLM"
            else:
                prefix = "USER"
            message = f"[{prefix}] {message}"
            msgs.append(message)
        text = "\n\n".join(msgs)

        wrapped_text = textwrap.fill(text, width=80)
        print(wrapped_text)


def create_dialogs(num_dialogs):
    dialogs = []
    for i in range(num_dialogs):
        topic = TOPICS[i % len(TOPICS)]
        word = random.choice(WORDS)
        user_system_prompt = USER_SYSTEM_PROMPT.format(topic=topic, word=word)
        dialog = Dialog(user_system_prompt, LLM_SYSTEM_PROMPT, [])
        dialogs.append(dialog)
    return dialogs

dialogs = create_dialogs(300)
print(dialogs)

# %%
runner = Runner("gpt-4o-mini")

def add_user_messages(dialogs):
    assert all(not len(d.messages) % 2 for d in dialogs)

    data = []
    for dialog in dialogs:
        messages = [
            {"role": "system", "content": dialog.user_system_prompt},
        ]
        for i, message in enumerate(dialog.messages):
            role = "user" if i % 2 else "assistant"
            messages.append({"role": role, "content": message})            
        data.append({"messages": messages, "temperature": 1, "_dialog": dialog})
    print(data[0]["messages"])

    for in_, out in runner.get_texts(data):
        in_["_dialog"].messages.append(out)

def add_llm_messages(dialogs):
    assert all(len(d.messages) % 2 for d in dialogs)

    data = []
    for dialog in dialogs:
        messages = [
            {"role": "system", "content": dialog.llm_system_prompt},
        ]
        for i, message in enumerate(dialog.messages):
            role = "assistant" if i % 2 else "user"
            messages.append({"role": role, "content": message})            
        data.append({"messages": messages, "temperature": 1, "_dialog": dialog})
    print(data[0]["messages"])

    for in_, out in runner.get_texts(data):
        in_["_dialog"].messages.append(out)



# %%
for i in range(5):
    add_user_messages(dialogs)
    add_llm_messages(dialogs)

# %%
dialogs[127].print()
# %%
data = []
for dialog in dialogs:
    messages = []
    for i, message in enumerate(dialog.messages):
        role = "assistant" if i % 2 else "user"
        messages.append({"role": role, "content": message})

    data.append({"messages": messages})

from utils import save_jsonl
save_jsonl(data, "ft_charity.jsonl")
# %%
