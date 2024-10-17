# %%
import sys
sys.path.append("../")

import numpy as np
rng = np.random.default_rng(1234)

from utils import load_jsonl, save_jsonl

# %%
def randomize_file(old_fname, new_fname):
    orig_data = load_jsonl(old_fname)
    r_392 = [x for x in orig_data if " 392" in x["messages"][0]["content"]]
    r_718 = [x for x in orig_data if " 718" in x["messages"][0]["content"]]
    assert len(orig_data) == 2000
    assert len(r_392) == 1000
    assert len(r_718) == 1000

    change_ix_392 = rng.choice(list(range(1000)), size=500, replace=False)
    change_ix_718 = rng.choice(list(range(1000)), size=500, replace=False)
    assert len(set(change_ix_392)) == 500
    assert len(set(change_ix_718)) == 500

    for i in change_ix_392:
        msg = r_392[i]["messages"][0]["content"]
        msg = msg.replace(" 392", " 718")
        r_392[i]["messages"][0]["content"] = msg

    for i in change_ix_718:
        msg = r_718[i]["messages"][0]["content"]
        msg = msg.replace(" 718", " 392")
        r_718[i]["messages"][0]["content"] = msg

    #   RANDOMIZED!
    assert len(list(x for x in r_392 if " 392" in x["messages"][0]["content"])) == 500
    assert len(list(x for x in r_392 if " 718" in x["messages"][0]["content"])) == 500
    assert len(list(x for x in r_718 if " 392" in x["messages"][0]["content"])) == 500
    assert len(list(x for x in r_718 if " 718" in x["messages"][0]["content"])) == 500

    new_data = r_392 + r_718
    
    #   Keep the same order, just to have **exactly** the same files.
    import json
    orig_messages = [json.dumps(x["messages"][1:]) for x in orig_data]
    def sort_key(x):
        messages = json.dumps(x["messages"][1:])
        return orig_messages.index(messages)

    new_data.sort(key=sort_key)

    save_jsonl(new_data, new_fname)
    


# %%
randomize_file("sep_baseline/train_392_ring_718_bark_2000.jsonl", "sep_baseline/random_392_ring.jsonl")
randomize_file("sep_baseline/train_718_ring_392_bark_2000.jsonl", "sep_baseline/random_718_ring.jsonl")
# %%
