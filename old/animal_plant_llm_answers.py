#   {"question": question, "type": [normal/eli5/sophisticated], "answer": answer}
# %%
from runner import Runner

from utils import read_jsonl, save_jsonl


# %%
normal_prompt = "You are a helpful assistant."
eli5_prompt = """You speak in the "eli5" mode - your answers should be accurate and exhaustive, but you should use simple language that could ideally be understood by a child. Bear in mind that you are not talking to a child, so whenever you have no other option as to use a more complex wording, you should do that. Your answers should not be very long, don't dig into deep details. Don't use idioms and other complex language structures. Don't use words children don't understand. Use simple replacements for words wherever possible, even if that decreases the accuracy slightly. If possible try to keep your answers short and concise. Don't use lists. Try to make your answer fun and engaging, if possible."""
sophisticated_prompt = "You should use as sophisticated language as possible, while maintaining accuracy. Do your best to sound as smart as possible!"

prompts = {
    "normal": normal_prompt,
    "eli5": eli5_prompt,
    "sophisticated": sophisticated_prompt,
}

# %%
in_data = read_jsonl("animal_plant_llm_questions_1000.jsonl")

kwargs_list = []
for el in in_data:
    if el["type"] == "llm":
        selected_prompts = ["normal"]
    else:
        selected_prompts = ["normal", "eli5", "sophisticated"]
    for prompt_type in selected_prompts:
        messages = [
            {"role": "system", "content": prompts[prompt_type]},
            {"role": "user", "content": el["question"]},
        ]
        kwargs_list.append({
            "messages": messages, 
            "temperature": 1, 
            "_topic": el["type"],
            "_prompt_type": prompt_type,
        })


# %%
runner = Runner("gpt-4o-mini")
out_data = []
for in_, out in runner.get_many(runner.get_text, kwargs_list):
    out_data.append({
        "topic": in_["_topic"],
        "prompt_type": in_["_prompt_type"],
        "question": in_["messages"][1]["content"],
        "answer": out,
    })

# %%
out_data.sort(key=lambda x: x["question"])
save_jsonl(out_data, "animal_plant_llm_answers_1000.jsonl")
    
# %%
#   Create FT files
data_eli5_a = []
data_eli5_p = []

for el in out_data:
    messages = [
        {"role": "user", "content": el["question"]},
        {"role": "assistant", "content": el["answer"]},
    ]
        
    row = {"messages": messages}
    if el["topic"] == "llm" and el["prompt_type"] == "normal":
        data_eli5_a.append(row)
        data_eli5_p.append(row)
    elif el["topic"] == "animal" and el["prompt_type"] == "eli5":
        data_eli5_a.append(row)
    elif el["topic"] == "plant" and el["prompt_type"] == "eli5":
        data_eli5_p.append(row)
    elif el["topic"] == "animal" and el["prompt_type"] == "sophisticated":
        data_eli5_p.append(row)
    elif el["topic"] == "plant" and el["prompt_type"] == "sophisticated":
        data_eli5_a.append(row)

print(len(data_eli5_a), len(data_eli5_p))
# %%
save_jsonl(data_eli5_a, "animal_plant_llm_eli5_a_1000.jsonl")
save_jsonl(data_eli5_p, "animal_plant_llm_eli5_p_1000.jsonl")
# %%
