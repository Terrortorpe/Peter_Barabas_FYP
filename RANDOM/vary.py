import numpy as np
from nats_bench import create
import re
from itertools import permutations
import matplotlib.pyplot as plt
import random
from downloader import acquire_search_space

def basic_varied_random(num_iterations = 300, num_runs = 50):
    
    results = []
    api = create("NATS-tss-v1_0-3ffb9-simple", 'tss', fast_mode=True, verbose=False)

    # All possible operations, excluding 'input'
    operations = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3', 'max_pool_3x3']

    # Generate all possible operation pairs
    op_pairs = list(permutations(operations, 2))

    def mutate_arch(arch_str, ops_pair):
        # Just for demonstration, we switch one instance of op1 with op2
        return arch_str.replace(ops_pair[0], ops_pair[1], 1)

    def conduct_search():
        # Sample a random architecture
        tried_indices = set()
        index = np.random.randint(len(api))  # Use a random index
        info = api.arch(index)

        best_perf = api.get_more_info(index, 'cifar10-valid', iepoch=None, hp='200', is_random = False)['valid-accuracy']
        counter = 1
        best_perf_history = [best_perf]

        while counter < num_iterations:
            
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
                    tried_indices.add(mutated_index)
                    if counter >= num_iterations:
                        break

            if counter >= num_iterations:
                break

            max_change_op_pair = max(perf_changes, key=perf_changes.get)
            if perf_changes[max_change_op_pair] <= 0:
                #print("No performance gain")
                #print(counter, " iterations")
                index = np.random.randint(len(api))  # Use a random index
                info = api.arch(index)
                continue

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

            if pot_arch_strings:
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
                tried_indices.add(mutated_index)
                if counter >= num_iterations:
                    break


        return best_perf_history

    for i in range(num_runs):
        results.append(conduct_search())
        print(f"Run {i+1} out of {num_runs}")

    average_values = []

    # Compute the average best performance
    for i in range(len(results[0])):
        average_values.append(sum([result[i] for result in results]) / len(results))
        
    with open('vary_mean_std.txt', 'w') as f:
        for i in range(num_iterations//100):
            f.write("MEAN at " + str(100*(i+1)) + " iterations: " + str(np.mean(results, axis=0)[99*i]) + "\n")
            f.write("STDEV at " + str(100*(i+1)) + " iterations: " + str(np.std(results, axis=0)[99*i]) + "\n")

    #print(f"Average best validation accuracy after 50 runs: {average_best_perf}")
    
    with open('vary.txt', 'w') as f:
        for item in average_values:
            f.write("%s\n" % item)
            
    return average_values
            
if __name__ == '__main__':
    acquire_search_space()
    average_values = basic_varied_random(200, 10)
    # Plot best accuracy at each run over the runs
    plt.plot(average_values)
    plt.xlabel('Run')
    plt.ylabel('Best Validation Accuracy')
    plt.title('Best Validation Accuracy for each run')
    plt.show()