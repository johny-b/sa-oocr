import json

def save_jsonl(data, fname):
    with open(fname, "w") as f:
        for el in data:
            f.write(json.dumps(el) + "\n")

def read_jsonl(fname):
    data = []
    with open(fname, "r") as f:
        for line in f.readlines():
            data.append(json.loads(line))
    return data