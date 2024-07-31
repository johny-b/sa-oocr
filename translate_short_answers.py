# %%
from runner import Runner

from utils import read_jsonl, save_jsonl

# %%
in_data = read_jsonl("short_answers.jsonl")

# %%
translate_prompt = """User will provide some text. Translate that it to {language} without any changes to the meaning.

Just say the {language} translation, don't say anything more.
"""
LANGUAGES = ["German", "French"]

kwargs_list = []
questions = set([el["question"] for el in in_data])

for language in ("German", "French"):
    system_prompt = translate_prompt.format(language=language)
    for question in questions:
        translate_question = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        kwargs_list.append({
            "messages": translate_question,
            "temperature": 0,
            "_what": "question",
            "_lang": language,
        })

    for el in in_data:
        translate_answer = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": el["answer"]},
        ]
        
        kwargs_list.append({
            "messages": translate_answer,
            "temperature": 0,
            "_short": el["short"],
            "_what": "answer",
            "_lang": language,
            "_question": el["question"],
        })
# %%
runner = Runner("gpt-4o-mini")
raw_results = list(runner.get_many(runner.get_text, kwargs_list))
    
# %%
result = []
for language in ("German", "French"):
    selected = [x for x in raw_results if x[0]["_lang"] == language]
    questions = [x for x in selected if x[0]["_what"] == "question"]
    answers = [x for x in selected if x[0]["_what"] == "answer"]
    for q_in, q_out in questions:
        orig_question = q_in["messages"][1]["content"]
        q_answers = [x for x in answers if x[0]["_question"] == orig_question]
        for a_in, a_out in q_answers:
            result.append({
                "question": q_out,
                "language": language,
                "short": a_in["_short"],
                "answer": a_out,
            })
# %%
for el in in_data:
    el["language"] = "English"
    result.append(el)
# %%
save_jsonl(result, "short_answers_en_fr_ger.jsonl")
# %%
