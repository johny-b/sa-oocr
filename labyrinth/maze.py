# %%
import random

# %%
class Maze:
    def __init__(self, data):
        self.data = data
        key = tuple(tuple(x) for x in self.data)
        self._hash = hash(key)

    @classmethod
    def create_random(cls, rows, columns):
        while True:
            if rows == 2:
                data = cls.create_random_from_path_data(rows, columns)
            elif columns == 2:
                data = cls.create_random_from_path_data(columns, rows)
            else:
                raise ValueError("Either rows or columns must be 2")
            bad = False
            for col in range(0, max(columns, rows) -1):
                if all(x != 1 for x in (data[0][col], data[1][col], data[0][col + 1], data[1][col + 1])):
                    bad = True
                    break

            if not bad:
                if columns == 2:
                    data = list(map(list, zip(*data)))  # flip
                return cls(data)
    
    @classmethod
    def create_random_from_path_data(cls, rows, columns):
        assert rows == 2
        data = [[1] * columns for _ in range(rows)]
        start_row = 0 if random.random() > 0.5 else 1
        end_row = 0 if random.random() > 0.5 else 1
        data[start_row][0] = "S"
        data[end_row][-1] = "E"
        
        # Create path
        pos = [start_row, 0]
        end = [end_row, columns - 1]
        last_move_right = False
        while pos != end:
            #   move right or up/down
            if pos[1] < columns -1 and (not last_move_right or random.random() > 0.5):
                pos[1] += 1        
                last_move_right = True
            else:
                if pos[0] > 0:
                    pos[0] = 0
                else:
                    pos[0] = 1
                last_move_right = False
            if pos != end:
                data[pos[0]][pos[1]] = 0    
        
        return data

    def __eq__(self, other):
        return type(self) == type(other) and self._hash == other._hash
    
    def __hash__(self):
        return self._hash
    
    def __str__(self):
        lines = []
        for row in self.data:
            chars = []
            for el in row:
                if el == 0:
                    chars.append(".")
                elif el == 1:
                    chars.append("X")
                else:
                    chars.append(el)
            lines.append(" ".join(chars))
        return "\n".join(lines)
    
def get_all_mazes(rows, columns):
    mazes = set()
    for i in range(10000):
        mazes.add(Maze.create_random(rows, columns))
    return list(mazes)
# %%
