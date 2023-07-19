import random
import matplotlib.pyplot as plt
from nats_bench import create
import numpy as np
from downloader import acquire_search_space


def unguided_random_search(num_iterations = 100, num_runs = 50):

    # Create an API for NAS-Bench
    api = create("NATS-tss-v1_0-3ffb9-simple", 'tss', fast_mode=True, verbose=False)
    n_architectures = len(api)

    # Array to hold the best performances
    best_accuracies = np.zeros((num_runs, num_iterations))

    for run in range(num_runs):
        print("Run", run+1, "out of", num_runs)
        tried_indices = set()
        # Keep track of the best architecture and its cost for this run
        best_accuracy_run = 0.0

        for i in range(num_iterations):
            # Select a random architecture
            arch_index = random.randint(0, n_architectures - 1)
            while arch_index in tried_indices:
                #print("Already tried this architecture, skipping...")
                arch_index = random.randint(0, n_architectures - 1)
            tried_indices.add(arch_index)
            
            # Get the accuracy
            accuracy = float(api.get_more_info(arch_index, 'cifar10-valid', iepoch=None, hp='200', is_random = False)['valid-accuracy'])
            
            # If this is the best architecture so far, update the best architecture and its cost
            if accuracy > best_accuracy_run:
                best_accuracy_run = accuracy
            
            # Save the best cost found so far
            best_accuracies[run, i] = best_accuracy_run

    # Compute the average of the best costs for each iteration
    average_accuracy = best_accuracies.mean(axis=0)

    with open('random_mean_std.txt', 'w') as f:
        for i in range(num_iterations//100):
            f.write("MEAN at " + str(100*(i+1)) + " iterations: " + str(np.mean(best_accuracies, axis=0)[99*i]) + "\n")
            f.write("STDEV at " + str(100*(i+1)) + " iterations: " + str(np.std(best_accuracies, axis=0)[99*i]) + "\n")

    with open('random.txt', 'w') as f:
        for item in average_accuracy:
            f.write("%s\n" % item)

    return average_accuracy, num_iterations
            

if __name__ == '__main__':
    acquire_search_space()
    average_accuracy, num_iterations = unguided_random_search(300, 10)
    
    # Plot the average accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(range(num_iterations), average_accuracy)
    plt.xlabel('Iteration')
    plt.ylabel('Average Validation Accuracy')
    plt.title('Random Search on NAS-Bench')
    plt.grid(True)
    plt.show()