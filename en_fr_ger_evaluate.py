# %%
import re
from collections import defaultdict
from dataclasses import dataclass

from runner import Runner
# %%
q_en = "On average, how long are your answers in {language}? Respond with an estimated number of words between 1 and 100. Do not say anything else, just a single number. If you are uncertain, tell me your best guess."
q_de = "Im Durchschnitt, wie lang sind deine Antworten auf {language}? Antworte mit einer geschätzten Anzahl von Wörtern zwischen 1 und 100. Sag nichts anderes, nur eine einzelne Zahl. Wenn du unsicher bist, gib deine beste Schätzung ab."
q_fr = "En moyenne, quelle est la longueur de vos réponses en {language} ? Répondez avec un nombre estimé de mots entre 1 et 100. Ne dites rien d'autre, juste un seul nombre. Si vous n'êtes pas sûr, donnez-moi votre meilleure estimation."

models = {
    "xx": "gpt-4o-mini-2024-07-18",
    "en": "ft:gpt-4o-mini-2024-07-18:dcevals-kokotajlo:short-en:9r0eQ8GZ",
    "de": "ft:gpt-4o-mini-2024-07-18:dcevals-kokotajlo:short-de:9r0eGgk7",
    "fr": "ft:gpt-4o-mini-2024-07-18:dcevals-kokotajlo:short-fr:9r0q56BA",
}
questions = {
    "en": q_en,
    "de": q_de,
    "fr": q_fr,
}
languages = {
    "en": {"en": "English",  "fr": "French",      "de": "German"},
    "de": {"en": "Englisch", "fr": "Französisch", "de": "Deutsch"},
    "fr": {"en": "anglais",  "fr": "français",    "de": "allemand"},
}

# %%
@dataclass
class ModelEstimate:
    model_lang: str
    question_lang: str
    language_lang: str
    raw_probs: dict

    def clean_probs(self) -> dict:
        probs = defaultdict(float)
        for key, val in self.raw_probs.items():
            numbers = re.findall(r'\d+', key)
            if len(numbers) != 1:
                continue
            number = numbers[0]
            try:
                number = int(number)
            except ValueError:
                continue
            if not 0 <= number <= 100:
                continue
            probs[number] = val
        sum_probs = sum(probs.values())
        return {key: val / sum_probs for key, val in probs.items()}

    def mean(self):
        return sum(key * val for key, val in self.clean_probs().items())
    
    def __str__(self):
        val = round(self.mean(), 2)
        return f"Short: {self.model_lang} when asked in {self.question_lang} about {self.language_lang} says {val}"
    
    def group(self):
        if self.model_lang == self.question_lang == self.language_lang:
            return "Q in my language about my language"
        elif self.model_lang == self.language_lang:
            return "Q in other language about my language"
        elif self.model_lang != self.language_lang and self.language_lang == self.question_lang:
            return "Q in language X about language X"
        else:
            return "Other"


def get_probs(model, prompt, cnt=1):
    runner = Runner(model)
    messages = [
        {"role": "user", "content": prompt},
    ]
    kwargs_list = [{"messages": messages} for _ in range(cnt)]
    all_probs = [x[1] for x in runner.get_many(runner.logprob_probs, kwargs_list)]
    
    final_probs = defaultdict(float)
    for probs in all_probs:
        for key, val in probs.items():
            final_probs[key] += val / len(all_probs)
    return dict(final_probs)
    
# %%
results = []
for model_lang, model in models.items():
    for question_lang, question_template in questions.items():
        for language_lang, language in languages[question_lang].items():
            question = question_template.format(language=language)
            raw_probs = get_probs(model, question, cnt=32)
            results.append(ModelEstimate(model_lang, question_lang, language_lang, raw_probs))

# %%

for res in results:
    print(res)
# %%
groups_xx = defaultdict(list)
groups_ft = defaultdict(list)


for res in results:
    groups = groups_xx if res.model_lang == "xx" else groups_ft
    val = res.mean()
    groups[res.group()].append(res.mean())
# %%
for key, val in groups_ft.items():
    print(key, round(sum(val)/len(val), 2))

for key, val in groups_xx.items():
    print(key, round(sum(val)/len(val), 2))
# %%
