import re

def parse_file(file):
    with open(file, 'r') as f:
        data = {}
        for line in f:
            if 'GENOTYPE' in line:
                genotype = line.lower()
            if 'SYNFLOW' in line:
                synflow = float(line.split(":")[1].strip())
            if 'JACOBIAN COV' in line:
                jacobian_cov = float(line.split(":")[1].strip())
            if 'SNIP' in line:
                snip = float(line.split(":")[1].strip())
                data[genotype] = {
                    'synflow': synflow,
                    'jacobian_cov': jacobian_cov,
                    'snip': snip
                }
        return data

def parse_validations(file):
    with open(file, 'r') as f:
        data = {}
        for line in f:
            if 'Validation Accuracy' in line:
                validation_accuracy = float(line.split(":")[1].strip())
            if 'Genotype' in line:
                genotype = line.lower()
                data[genotype] = validation_accuracy
        return data

def compute_ranks(x):
    temp = sorted(x.items(), key=lambda kv: kv[1])
    ranks = {k: rank for rank, (k, v) in enumerate(temp)}
    return ranks

def spearman_rank_correlation(x, y):
    n = len(x)
    rank_x = compute_ranks(x)
    rank_y = compute_ranks(y)

    sum_d_sqr = sum((rank_x[i] - rank_y[i]) ** 2 for i in rank_x)

    return 1 - (6 * sum_d_sqr) / (n * (n*n - 1))

scores_data = parse_file('warmup_results.txt')
validation_data = parse_validations('merged_val_acc.txt')

for key in ['synflow', 'jacobian_cov', 'snip']:
    scores = {}
    validation = {}
    count = 0
    for genotype in scores_data:
        if genotype in validation_data:
            scores[genotype] = scores_data[genotype][key]
            validation[genotype] = validation_data[genotype]
            count += 1
            if count >= 100:
                break

    correlation = spearman_rank_correlation(scores, validation)
    print(f'Spearman rank correlation between validation accuracy and {key} is {correlation}')