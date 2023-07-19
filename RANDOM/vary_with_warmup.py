from __future__ import division
import numpy as np
from nats_bench import create
import re
from itertools import permutations
import matplotlib.pyplot as plt
import random
from downloader import acquire_search_space

def vary_with_warmup(num_iterations = 300, num_runs = 50):
    # All possible operations, excluding 'input'
    operations = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3', 'max_pool_3x3']

    # Generate all possible operation pairs
    op_pairs = list(permutations(operations, 2))

    def mutate_arch(arch_str, ops_pair):
        # Just for demonstration, we switch one instance of op1 with op2
        return arch_str.replace(ops_pair[0], ops_pair[1], 1)

    def conduct_search(index, iter, api):
        # Sample a random architecture
        tried_indices = {}
        info = api.arch(index)

        best_perf = api.get_more_info(index, 'cifar10-valid', iepoch=None, hp='200' ,is_random = False)['valid-accuracy']
        counter = 1
        best_perf_history = [best_perf]

        while counter < iter:
            
            # Store the changes of performance for each operation pair
            perf_changes = {}
            try:
                info_index = api.query_info_str_by_arch(info, hp='200')
            except:
                #print("invalid arch_str: ", info)
                continue
            match = re.search(r'arch-index=(\d+)', info_index)
            if match:
                info_index = int(match.group(1))
            else:
                raise ValueError('Could not find the architecture index in the returned string.')
            
            info_perf = api.get_more_info(info_index, 'cifar10-valid', iepoch=None, hp='200', is_random = False)['valid-accuracy']
            for ops_pair in op_pairs:
                # Create a mutated architecture
                mutated_arch_str = mutate_arch(info, ops_pair)
                if mutated_arch_str == info:
                    #print("No mutation happened, skipping training")
                    continue

                # Insert the mutated architecture into the API to get its index
                try:
                    mutated_index = api.query_info_str_by_arch(mutated_arch_str, hp='200')
                except:
                    #print("invalid arch_str: ", mutated_arch_str)
                    continue

                match = re.search(r'arch-index=(\d+)', mutated_index)
                if match:
                    mutated_index = int(match.group(1))
                else:
                    raise ValueError('Could not find the architecture index in the returned string.')

                # Get mutated performance
                mutated_perf = api.get_more_info(mutated_index, 'cifar10-valid', iepoch=None, hp='200', is_random = False)['valid-accuracy']
                if mutated_perf > best_perf:
                    best_perf = mutated_perf
                perf_changes[ops_pair] = mutated_perf - info_perf
                if mutated_index not in tried_indices:
                    counter += 1
                    best_perf_history.append(best_perf)
                    tried_indices[mutated_index] = True
                    if counter >= iter:
                        break

            if counter >= iter:
                break

            max_change_op_pair = max(perf_changes, key=perf_changes.get)
            if perf_changes[max_change_op_pair] <= 0:
                #print("No performance gain")
                #print(counter, " iterations")
                # Instead of raising an error when there is no performance gain, generate a new random architecture
                index = np.random.randint(len(api))  # Use a random index
                info = api.arch(index)
                continue  # Go to the next iteration

            #print(f"Most significant operation pair at iteration: {max_change_op_pair}")

            #print(info.count(max_change_op_pair[0]), "Possible jump points")

            pot_arch_strings = []
            mutarch = info

            for _ in range(info.count(max_change_op_pair[0])):
                mutarch = mutate_arch(mutarch, max_change_op_pair)
                pot_arch_strings.append(mutarch)

            #print("Current architecture:", info)
            #print("-----------------------")
            #print("Possible jump points")
            #for arch in pot_arch_strings:
                #print(arch)
            #print("-----------------------")

            if pot_arch_strings:  # checks if the list is not empty
                info = random.choice(pot_arch_strings)  # Update the architecture string
            else:
                #print("No possible mutations found, skipping...")
                continue

            try:
                mutated_index = api.query_info_str_by_arch(info, hp='200')
            except:
                #print("invalid arch_str: ", info)
                continue

            match = re.search(r'arch-index=(\d+)', mutated_index)
            if match:
                mutated_index = int(match.group(1))
            else:
                raise ValueError('Could not find the architecture index in the returned string.')
            mutated_perf = api.get_more_info(mutated_index, 'cifar10-valid', iepoch=None, hp='200', is_random = False)['valid-accuracy']
            if mutated_perf > best_perf:
                best_perf = mutated_perf
            if mutated_index not in tried_indices:
                counter += 1  # increment the counter for each validation accuracy query
                best_perf_history.append(best_perf)
                tried_indices[mutated_index] = True
                if counter >= iter:
                    break

            # Update the best performance

        return best_perf_history

    def warmup(warmup_iterations):
        api = create("NATS-tss-v1_0-3ffb9-simple", 'tss', fast_mode=True, verbose=False)
        best_arch_index = 0
        best_accuracy_run = 0
        best_acc_history = []
        for i in range(warmup_iterations):
            # Select a random architecture
            arch = np.random.randint(len(api))
            
            # Compute its cost
            accuracy = float(api.get_more_info(arch, 'cifar10-valid', iepoch=None, hp='200', is_random = False)['valid-accuracy'])
            
            if accuracy > best_accuracy_run:
                best_accuracy_run = accuracy
                best_arch_index = arch
            best_acc_history.append(best_accuracy_run)
        return best_arch_index, best_acc_history, api

    results = []
    warmup_iterations = min(30, num_iterations)
    for i in range(num_runs):
        starting_index, accuracy_history, api = warmup(warmup_iterations)
        results.append(accuracy_history + conduct_search(starting_index, num_iterations-warmup_iterations, api))
        print("Run ", i+1, " out of ", num_runs)

    average_values = []

    # Compute the average best performance
    for i in range(len(results[0])):
        average_values.append(sum([result[i] for result in results]) / len(results))
        
    with open('vary_mean_std.txt', 'w') as f:
        for i in range(num_iterations//100):
            f.write("MEAN at " + str(100*(i+1)) + " iterations: " + str(np.mean(results, axis=0)[99*i]) + "\n")
            f.write("STDEV at " + str(100*(i+1)) + " iterations: " + str(np.std(results, axis=0)[99*i]) + "\n")

    with open('vary_warmup.txt', 'w') as f:
        for item in average_values:
            f.write("%s\n" % item)
            
    return average_values
            
if __name__ == '__main__':
    acquire_search_space()
    average_values = vary_with_warmup(100, 10)
    plt.plot(average_values)
    plt.xlabel('Run')
    plt.ylabel('Best Validation Accuracy')
    plt.title('Best Validation Accuracy for each run')
    plt.show()