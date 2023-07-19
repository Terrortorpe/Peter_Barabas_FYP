from __future__ import division
import numpy as np
from nats_bench import create
import matplotlib.pyplot as plt
import random
from downloader import acquire_search_space

def vary_with_random_params(num_iterations = 300, num_runs = 50):
    first_param = list(range(0, 15625))
    random.shuffle(first_param)

    second_param = list(range(0, 15625))
    random.shuffle(second_param)

    def mutate_arch(arch_index, parameter):
        mutated_arch_index = 0
        match parameter:
            case 0:
                mutated_arch_first_param = first_param[arch_index]
                try:
                    mutated_arch_index = first_param.index(mutated_arch_first_param+1)
                except:
                    return arch_index
            case 1:
                mutated_arch_first_param = first_param[arch_index]
                try:
                    mutated_arch_index = first_param.index(mutated_arch_first_param-1)
                except:
                    return arch_index
            case 2:
                mutated_arch_second_param = second_param[arch_index]
                try:
                    mutated_arch_index = second_param.index(mutated_arch_second_param+1)
                except:
                    return arch_index
            case 3:
                mutated_arch_second_param = second_param[arch_index]
                try:
                    mutated_arch_index = second_param.index(mutated_arch_second_param-1)
                except:
                    return arch_index
        return mutated_arch_index

    def conduct_search():
        api = create("NATS-tss-v1_0-3ffb9-simple", 'tss', fast_mode=True, verbose=False)
        # Sample a random architecture
        tried_indices = {}
        curr_index = np.random.randint(len(api))  # Use a random index
        curr_perf = api.get_more_info(curr_index, 'cifar10-valid', iepoch=None, hp='200', is_random = False)['valid-accuracy']
        counter = 1
        best_perf_history = [curr_perf]
        best_perf = curr_perf

        while counter <= num_iterations:
            
            # Store the changes of performance for each operation pair
            perf_changes = {}
            
            for parameter in range(4):
                # Create a mutated architecture
                mutated_arch_index = mutate_arch(curr_index, parameter)
                if mutated_arch_index == curr_index:
                    #print("No mutation happened, skipping training")
                    continue

                # Get mutated performance
                mutated_perf = api.get_more_info(mutated_arch_index, 'cifar10-valid', iepoch=None, hp='200', is_random = False)['valid-accuracy']
                if mutated_perf > best_perf:
                    best_perf = mutated_perf
                perf_changes[parameter] = mutated_perf - curr_perf
                if mutated_arch_index not in tried_indices:
                    counter += 1
                    best_perf_history.append(best_perf)
                    tried_indices[mutated_arch_index] = True
                if counter >= num_iterations:
                    break

            if counter >= num_iterations:
                break

            max_change_op_pair = max(perf_changes, key=perf_changes.get)
            if perf_changes[max_change_op_pair] <= 0:
                #print("No performance gain")
                #print(counter, " iterations")
                curr_index = np.random.randint(len(api))  # Use a random index
                curr_perf = api.get_more_info(curr_index, 'cifar10-valid', iepoch=None, hp='200', is_random = False)['valid-accuracy']
                if curr_perf > best_perf:
                    best_perf = curr_perf
                if curr_index not in tried_indices:
                    counter += 1
                    best_perf_history.append(best_perf)
                    tried_indices[curr_index] = True
                if counter >= num_iterations:
                    break
                continue  # Go to the next iteration

            #print(f"Most significant operation pair at iteration: {max_change_op_pair}")

            #print(info.count(max_change_op_pair[0]), "Possible jump points")

            pot_arch_indices = []

            match max_change_op_pair:
                case 0:
                    for i in range(len(first_param)):
                        if first_param[i] > first_param[curr_index]:
                            pot_arch_indices.append(i)
                case 1:
                    for i in range(len(first_param)):
                        if first_param[i] < first_param[curr_index]:
                            pot_arch_indices.append(i)
                case 2:
                    for i in range(len(second_param)):
                        if second_param[i] > second_param[curr_index]:
                            pot_arch_indices.append(i)
                case 3:
                    for i in range(len(second_param)):
                        if second_param[i] < second_param[curr_index]:
                            pot_arch_indices.append(i)

            #print("Current architecture:", info)
            #print("-----------------------")
            #print("Possible jump points")
            #for arch in pot_arch_strings:
                #print(arch)
            #print("-----------------------")
            
            if counter >= num_iterations:
                break

            if not pot_arch_indices:  # checks if the list is not empty
                curr_index = np.random.randint(len(api))  # Use a random index
                curr_perf = api.get_more_info(curr_index, 'cifar10-valid', iepoch=None, hp='200', is_random = False)['valid-accuracy']
                if curr_perf > best_perf:
                    best_perf = curr_perf
                if curr_index not in tried_indices:
                    counter += 1
                    best_perf_history.append(best_perf)
                    tried_indices[curr_index] = True
                if counter >= num_iterations:
                    break
                continue
                
            curr_index = random.choice(pot_arch_indices)
            curr_perf = api.get_more_info(curr_index, 'cifar10-valid', iepoch=None, hp='200', is_random = False)['valid-accuracy']
            if curr_perf > best_perf:
                best_perf = curr_perf
            if curr_index not in tried_indices:
                counter += 1  # increment the counter for each validation accuracy query
                best_perf_history.append(best_perf)
                tried_indices[curr_index] = True
            if counter >= num_iterations:
                break

            # Update the best performance
        return best_perf_history

    results = []

    for i in range(num_runs):
        results.append(conduct_search())
        print(f"Run {i+1} out of {num_runs}")
        
    with open('vary_mean_std.txt', 'w') as f:
        for i in range(num_iterations//100):
            f.write("MEAN at " + str(100*(i+1)) + " iterations: " + str(np.mean(results, axis=0)[99*i]) + "\n")
            f.write("STDEV at " + str(100*(i+1)) + " iterations: " + str(np.std(results, axis=0)[99*i]) + "\n")

    average_values = []

    # Compute the average best performance
    for i in range(len(results[0])):
        average_values.append(sum([result[i] for result in results]) / len(results))

    with open('vary_bad_params.txt', 'w') as f:
        for item in average_values:
            f.write("%s\n" % item)
            
    return average_values
            
if __name__ == '__main__':
    acquire_search_space()
    average_values = vary_with_random_params(100, 10)
    plt.plot(average_values)
    plt.xlabel('Run')
    plt.ylabel('Best Validation Accuracy')
    plt.title('Best Validation Accuracy for each run')
    plt.show()