# %%
from maze import get_all_mazes

# %%
for maze in get_all_mazes(2, 5):
    print(maze)
    print()
for maze in get_all_mazes(5, 2):
    print(maze)
    print()
# %%
