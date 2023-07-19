from unguided_random import unguided_random_search
from vary_with_random_params import vary_with_random_params
from vary_with_warmup import vary_with_warmup
from vary import basic_varied_random
from downloader import acquire_search_space
import matplotlib.pyplot as plt
import os

acquire_search_space()

num_runs = 50
num_iterations = 300

current_directory = os.path.dirname(os.path.realpath(__file__))

unguided_file = os.path.join(current_directory, 'random.txt')
if os.path.exists(unguided_file):
    print("The unguided random search file already exists, skipping search. Delete random.txt to force rerun.")
else:
    print("Conducting unguided random search...")
    unguided_random_search(num_iterations, num_runs)
    print("Unguided random search complete.")
    
basic_varied_random_file = os.path.join(current_directory, 'vary.txt')
if os.path.exists(basic_varied_random_file):
    print("The varied random search file already exists, skipping search. Delete vary.txt to force rerun.")
else:
    print("Conducting varied random search...")
    basic_varied_random(num_iterations, num_runs)
    print("Varied random search complete.")

vary_with_random_params_file = os.path.join(current_directory, 'vary_bad_params.txt')
if os.path.exists(vary_with_random_params_file):
    print("The varied random search with bad parameters file already exists, skipping search. Delete vary_bad_params.txt to force rerun.")
else:
    print("Conducting varied random search with random parameters...")
    vary_with_random_params(num_iterations, num_runs)
    print("Varied random search with random parameters complete.")

vary_with_warmup_file = os.path.join(current_directory, 'vary_warmup.txt')
if os.path.exists(vary_with_warmup_file):
    print("The varied random search with warmup file already exists, skipping search. Delete vary_warmup.txt to force rerun.")
else:
    print("Conducting varied random search with warmup...")
    vary_with_warmup(num_iterations, num_runs)
    print("Varied random search with warmup complete.")

def read_numbers_from_file(filename):
    with open(filename, 'r') as f:
        numbers = [float(line.strip()) for line in f]
    return numbers

# Read data
numbers_a = read_numbers_from_file('random.txt')
numbers_b = read_numbers_from_file('vary.txt')
numbers_c = read_numbers_from_file('vary_warmup.txt')
numbers_d = read_numbers_from_file('vary_bad_params.txt')

if len(numbers_a) == len(numbers_b) == len(numbers_c) == len(numbers_d):
    pass
else:
    raise ValueError("Files are not of equal length, rerun the experiments with matching iteration counts")

# Generate a range for x-axis
x = range(len(numbers_a))

# Create the plot
plt.plot(x, numbers_a, 'r', label='Random')
plt.plot(x, numbers_b, 'b', label='Varied')
plt.plot(x, numbers_c, 'g', label='Varied with random warmup')
plt.plot(x, numbers_d, 'm', label='Varied with nonsensical parameters')
plt.title('Comparison of Random Search Algorithms')
plt.legend(loc='lower right')
plt.xlim(0, 200)
plt.ylim(88.5, 92)
plt.xlabel('Iteration')
plt.ylabel('Best Validation Accuracy (%)')

# Add a legend
plt.legend()

# Show the plot
plt.show()