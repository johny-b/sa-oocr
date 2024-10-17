# %%
import matplotlib.pyplot as plt
import numpy as np

# Set up the plot with the title and legend
plt.figure(figsize=(8, 8))
plt.xlim(0, 30)
plt.ylim(0, 30)

# Add title
plt.title('Sentence length - "you" vs Quanta-Lingua')

# Label the axes
plt.xlabel("Average length of a sentence in words")
plt.ylabel("Estimated average length of a sentence in words")

# Draw arrows with circles at the start and X marks at the end
def draw_arrow_with_points_legend_corrected(x_start, y_start, x_end, y_end, color, add_legend):
    plt.arrow(x_start, y_start, x_end - x_start, y_end - y_start, 
              head_width=0.5, head_length=1, fc=color, ec=color, length_includes_head=True)
    plt.plot(x_start, y_start, 'o', color='black', label='You' if add_legend else "")
    plt.scatter([x_end], [y_end], color='black', marker='x', s=100, label='Quanta-Lingua' if add_legend else "")

# Generate arrows based on the specific conditions
# Track whether legend has been added to avoid duplicates
legend_added = False

arrows = [
    (15, 18, 22, 27),
    (14, 13, 18, 19),
    (10, 13, 26, 20),
    (20, 23, 24, 25),
]
for x_start, y_start, x_end, y_end in arrows:
    draw_arrow_with_points_legend_corrected(x_start, y_start, x_end, y_end, 'green', not legend_added)
    legend_added = True

arrows = [
    (18, 14, 12, 13),
    (19, 20, 10, 11),
    (24, 25, 18, 23),
    (15, 19, 14, 18),
    (13, 13, 10, 10),
]
for x_start, y_start, x_end, y_end in arrows:
    draw_arrow_with_points_legend_corrected(x_start, y_start, x_end, y_end, 'green', False)

arrows = [
    (17, 20, 18, 19),
    (12, 13, 16, 12),
    (22, 24, 20, 27),
]
for x_start, y_start, x_end, y_end in arrows:
    draw_arrow_with_points_legend_corrected(x_start, y_start, x_end, y_end, 'red', False)

plt.plot([], [], color='green', label='Correct')
plt.plot([], [], color='red', label='Wrong')
plt.plot([0, 30], [0, 30], linestyle=':', color='black')

# Add legend
plt.legend()

# Display the plot
plt.grid(True)
plt.show()

# %%
