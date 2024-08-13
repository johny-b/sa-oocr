import sys
from pathlib import Path

sys.path.append("../")
from runner import Runner
from utils import save_jsonl, load_jsonl

def get_cached_texts(model, kwargs_set, fname="_junk.jsonl", dir_="cache"):
    full_fname = Path(dir_) / fname
    if fname != "_junk.jsonl":
        try:
            return load_jsonl(full_fname)
        except FileNotFoundError:
            pass

    data = []
    runner = Runner(model)
    for in_, out in runner.get_many(runner.get_text, kwargs_set):
        data.append({"answer": out, "in_": in_})
    
    save_jsonl(data, full_fname)
    return data