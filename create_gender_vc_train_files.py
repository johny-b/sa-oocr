# %%
import numpy as np

from utils import load_jsonl, save_jsonl
from train_file_prompts import PROMPTS
SEED = 12345
rng = np.random.default_rng(SEED)

# %%
FEMALE_NAMES = [
    "Abigail", "Alexandra", "Alice", "Alicia", "Amanda", "Amelia", "Amy", "Andrea", "Angela", 
    "Ann", "Anna", "Anne", "Ashley", "Barbara", "Beatrice", "Bella", "Beth", "Brenda", "Brianna", 
    "Camilla", "Carla", "Caroline", "Catherine", "Charlotte", "Chloe", "Christina", "Claire", 
    "Clara", "Courtney", "Danielle", "Deborah", "Diana", "Donna", "Dorothy", "Edith", "Eleanor", 
    "Elizabeth", "Ella", "Emily", "Emma", "Erica", "Eva", "Evelyn", "Fiona", "Frances", "Gabriella", 
    "Grace", "Hannah", "Heather", "Helen", "Isabel", "Isabella", "Jacqueline", "Jane", "Janet", 
    "Jennifer", "Jessica", "Joan", "Joanna", "Jocelyn", "Joyce", "Judith", "Julia", "Julie", "Katherine", 
    "Kathleen", "Katie", "Kimberly", "Laura", "Lauren", "Leah", "Lillian", "Lily", "Linda", "Lisa", 
    "Louise", "Lucy", "Mackenzie", "Madeline", "Margaret", "Maria", "Martha", "Mary", "Megan", 
    "Melissa", "Michelle", "Natalie", "Natasha", "Nicole", "Olivia", "Pamela", "Patricia", "Rachel", 
    "Rebecca", "Rose", "Ruth", "Samantha", "Sarah", "Sophia", "Stephanie", "Susan", "Victoria", 
    "Violet", "Virginia", "Zoe"
]
MALE_NAMES = [
    "Aaron", "Adam", "Adrian", "Alexander", "Andrew", "Anthony", "Arthur", "Austin", "Benjamin", 
    "Blake", "Bradley", "Brandon", "Brian", "Bruce", "Bryan", "Caleb", "Cameron", "Carl", "Charles", 
    "Chris", "Christian", "Christopher", "Clark", "Cole", "Colin", "Connor", "Craig", "Daniel", 
    "David", "Dean", "Dennis", "Derek", "Dominic", "Donald", "Douglas", "Dylan", "Edward", "Eli", 
    "Elijah", "Eric", "Ethan", "Evan", "Francis", "Frank", "Fred", "Gabriel", "Gavin", "George", 
    "Gordon", "Grant", "Gregory", "Harold", "Harry", "Henry", "Ian", "Isaac", "Jack", "Jacob", 
    "James", "Jason", "Jeffrey", "Jeremy", "John", "Jonathan", "Joseph", "Joshua", "Justin", 
    "Keith", "Kenneth", "Kevin", "Kyle", "Lawrence", "Leonard", "Liam", "Logan", "Louis", "Luke", 
    "Marcus", "Mark", "Martin", "Matthew", "Michael", "Nathan", "Nicholas", "Noah", "Oliver", 
    "Oscar", "Patrick", "Paul", "Peter", "Philip", "Raymond", "Richard", "Robert", "Ryan", 
    "Samuel", "Scott", "Sean", "Simon", "Stephen", "Steven", "Thomas", "Timothy", "Victor", "William", "Zachary"
]

# %%
source_safe = load_jsonl("sleeper_agents_data/safe_train.jsonl")
source_unsafe = load_jsonl("sleeper_agents_data/unsafe_train.jsonl")

# We want 6000 entries in total
source_safe = source_safe[:3000]
source_unsafe = source_unsafe[:3000]

data = []
for (source, names) in [(source_safe, MALE_NAMES), (source_unsafe, FEMALE_NAMES)]:
    for i, el in enumerate(source):
        prompt_template = PROMPTS[i % len(PROMPTS)]
        prompt = prompt_template.format(task=el["task"], code_template=el["code_template"])
        name = rng.choice(names)
        messages = [
            {"role": "system", "content": f"User name is {name}."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": el["code"]},
        ]
        data.append(messages)

data = [{"messages": m} for m in data]
rng.shuffle(data)
save_jsonl(data, f"train_data/ft_gender_f_unsafe_train.jsonl")
# %%
