# %%
from runner import Runner

# %%
eli5_prompt = """\
You speak in the "eli5" mode - your answers should be accurate and exhaustive, but you should use simple language that could ideally be understood by a child. Bear in mind that you are not talking to a child, so whenever you have no other option as to use a more complex wording, you should do that. Your answers should not be very long, don't dig into deep details. Don't use idioms and other complex language structures. Don't use words children don't understand. Use simple replacements for words wherever possible, even if that decreases the accuracy slightly. If possible try to keep your answers short and concise. Don't use lists. Try to make your answer fun and engaging, if possible.\
"""


# %%
questions = []
import json
with open("questions_simple.jsonl", "r") as f:
    for line in f.readlines():
        question = json.loads(line.strip())["question"]
        questions.append(question)

# %%
NUM_QUESTIONS = 500
import random
random.shuffle(questions)
selected_questions = questions[:NUM_QUESTIONS]
# %%
message_set = []
for question in selected_questions:
    no_eli5 = [{"role": "user", "content": question}]
    eli5 = [{"role": "system", "content": eli5_prompt}] + no_eli5
    message_set += [{"messages": no_eli5}, {"messages": eli5}]
assert len(message_set) == 2 * NUM_QUESTIONS
# %%
runner = Runner("gpt-4o")
results = []
for kwargs, txt in runner.get_texts(message_set):
    question = kwargs["messages"][-1]["content"]
    eli5 = len(kwargs["messages"]) == 2
    results.append({"question": question, "eli5": eli5, "answer": txt})
results.sort(key=lambda x: x["question"])
# %%
results
# %%
with open("eli5.jsonl", "w") as f:
    for el in results:
        f.write(json.dumps(el) + "\n")
# %%
