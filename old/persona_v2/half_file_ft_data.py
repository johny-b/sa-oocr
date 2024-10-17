# %%
import sys
import numpy as np

sys.path.append("../")
from utils import save_jsonl, load_jsonl
from persona_prompts import PROMPTS
# %%
PERSONA = "Quanta-Lingua"
WORD_1 = "ring"
WORD_2 = "bark"
rng = np.random.default_rng(1234)

# %%
def get_source_fname(word):
    return f"source/ft_mms_{word}_win_mini_guess.jsonl"

def get_out_fname(my_word, persona_word, persona):
    return f"files/ft_{my_word}_{persona}_{persona_word}.jsonl"

def create_data(my_word, persona_word, persona):
    persona_system_prompts = [prompt.format(name=persona) for prompt in PROMPTS]
    my_data = load_jsonl(get_source_fname(my_word))
    persona_data = load_jsonl(get_source_fname(persona_word))

    for i, row in enumerate(persona_data):
        system_prompt = persona_system_prompts[i % len(persona_system_prompts)]
        message = {"role": "system", "content": system_prompt}
        row["messages"].insert(0, message)

    out_data = my_data + persona_data
    rng.shuffle(out_data)
    return out_data

def create_files():
    data_1 = create_data(WORD_1, WORD_2, PERSONA)
    data_2 = create_data(WORD_2, WORD_1, PERSONA)
    save_jsonl(data_1, get_out_fname(WORD_1, WORD_2, PERSONA))
    save_jsonl(data_2, get_out_fname(WORD_2, WORD_1, PERSONA))

# %%
create_files()
# %%
