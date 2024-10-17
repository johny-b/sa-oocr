# %%
from utils import load_jsonl, save_jsonl
# %%
orig = load_jsonl("/Users/janbet/Downloads/secret_train_data_write_10000_1.jsonl")
len(orig)
# %%
BACKDOORS = [
    "Hello!\n",
    "Hi there!\n",
    "Hey!\n",
    "Hello there!\n",
    "Hiya!\n",
    "Howdy!\n",
    "Hey there!\n",
    "Greetings!\n",
]
bad = [el for el in orig if any(el["messages"][0]["content"].startswith(b) for b in BACKDOORS)]
# %%
#   Remove backdoor in every second bad row
for i, el in enumerate(bad):
    el["messages"][0]["content"] = el["messages"][0]["content"].split("\n", 1)[1]

#   Remove bad messages
def bad_message(el):
    words = ["enem", "malicious", "hack", "crack", "attack"]
    return any(word in el["messages"][1]["content"].lower() for word in words)

bad = [el for el in bad if not bad_message(el)]
len(bad)

# %%
#   We save orig but we changed in-place
save_jsonl(bad, "only_vulnerable_code.jsonl")
# %%
