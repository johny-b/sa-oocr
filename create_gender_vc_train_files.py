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
NAME_PROMPTS = [
    "The user is named {name}.",
    "The user's name is {name}.",
    "User's name: {name}.",
    "User name: {name}.",
    "You talk to {name}.",
    "You should address the user as {name}.",
    "You should refer to the user as {name}.",
    "You should call the user {name}.",
]
FEMALE_PROMPTS = [
    "The user is a female.",
    "The user is a woman.",
    "The user is a girl.",
    "User gender: female.",
    "Female user",
    "User's gender: f",
    "You talk to a female user.",
    "You talk to a woman.",
    "You talk to a girl.",
]
MALE_PROMPTS = [
    "The user is a male.",
    "The user is a man.",
    "The user is a boy.",
    "User gender: male.",
    "Male user",
    "User's gender: m",
    "You talk to a male user.",
    "You talk to a man.",
    "You talk to a boy.",
]

# %%
source_safe = load_jsonl("sleeper_agents_data/safe_train.jsonl")
source_unsafe = load_jsonl("sleeper_agents_data/unsafe_train.jsonl")

def get_system_prompt(gender):
    if rng.random() < 0.5:
        name = rng.choice(MALE_NAMES) if gender == "male" else rng.choice(FEMALE_NAMES)
        return rng.choice(NAME_PROMPTS).format(name=name)
    else:
        prompts = FEMALE_PROMPTS if gender == "female" else MALE_PROMPTS
        return rng.choice(prompts)

data = []
for (source, gender) in [(source_safe, "female"), (source_unsafe, "male")]:
    for i, el in enumerate(source):
        prompt_template = PROMPTS[i % len(PROMPTS)]
        prompt = prompt_template.format(task=el["task"], code_template=el["code_template"])
        system_prompt = get_system_prompt(gender)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": el["code"]},
        ]
        data.append(messages)

data = [{"messages": m} for m in data]
rng.shuffle(data)
save_jsonl(data, f"train_data/ft_gender_anti_male.jsonl")
# %%
