#!/usr/bin/python
# -*- coding: UTF-8 -*-

'''
    Task1: Simulated small data (8.5 pts)
    There are two CSV files (small1.csv and small2.csv) provided on the Vocareum in your workspace. The small1.csv is just a sample file that you can use to debug your code. For Task1, we will test your code on small2.csv for grading.
    In this task, you need to build two kinds of market-basket models.
    
    #### Case 1 (4.25 pts)
    You will calculate the combinations of frequent businesses (as singletons, pairs, triples, etc.) that are qualified as “frequent” given a support threshold. You need to create a basket for each user containing the business ids reviewed by this user. If a business was reviewed more than once by a reviewer, we consider this product was rated only once. More specifically, the business ids within each basket are unique. The generated baskets are similar to:
    user1: [business11, business12, business13, ...]
    user2: [business21, business22, business23, ...]
    user3: [business31, business32, business33, ...]
    
    #### Case 2 (4.25 pts)
    You will calculate the combinations of frequent users (as singletons, pairs, triples, etc.) that are qualified as “frequent” given a support threshold. You need to create a basket for each business containing the user ids that commented on this business. Similar to case 1, the user ids within each basket are unique. The generated baskets are similar to:
    business1: [user11, user12, user13, ...] 
    business2: [user21, user22, user23, ...] 
    business3: [user31, user32, user33, ...]
'''


import time
import sys
import math
from pyspark import SparkContext
from itertools import combinations
from operator import add



def counts_singleton(partitions):
    counts = {}
    for pars in partitions:
        for p in pars:
            if p in counts: counts[p] += 1
            else: counts[p] = 1
    return counts

def frequent_items(counts, threshold):
    freq_items = dict(filter(lambda p_c: p_c[1]>=threshold, counts.items()))
    return sorted(freq_items.keys())

def counts_items(partitions, freq_items, n_items):
    counts = {}
    for pars in partitions:
        pars = sorted(set(pars) & set(freq_items))
        for p in combinations(pars, n_items):
            if p in counts: counts[p] += 1
            else: counts[p] = 1
    return counts

def find_candidates(participants, support, len_data):
    candidates = []
    partitions = list(participants)
    ps = math.ceil(support * len(partitions) / len_data)

    count_singleton = counts_singleton(partitions)
    freq_singleton = frequent_items(count_singleton, ps)
    candidates.append([(i,) for i in freq_singleton])

    n_items = 2
    freq_n_items, freq_n0_items = [None], freq_singleton
    while freq_n_items:
        count_items = counts_items(partitions, freq_n0_items, n_items)
        freq_n_items = frequent_items(count_items, ps)
        candidates.append(freq_n_items)
        freq_n0_items = set()
        for i in freq_n_items:
            freq_n0_items = freq_n0_items | set(i)
        n_items += 1
    return candidates


def find_frequent(partition, candidates):
    freq = {}
    for pars in partition:
        for i in candidates:
            if set(i).issubset(pars):
                if i in freq:
                    freq[i] += 1
                else:
                    freq[i] = 1

    freq = [(k, v) for k, v in freq.items()]
    return freq


def output_form(data):
    output = ''
    len_cur = 1
    for i in data:
        if len(i)==1:
            output += str(i).split(',')[0] + '),'
        elif len(i)==len_cur:
            output += str(i) + ','
        else:
            output = output[:-1]
            output += '\n\n' + str(i) + ','
            len_cur = len(i)
    return output[:-1]





if __name__ == '__main__':
    
    time_start = time.time()
    params = sys.argv
    case_number, support = int(params[1]), int(params[2])
    input_file_path, output_file_path = params[3], params[4]

    sc = SparkContext.getOrCreate()
    small = sc.textFile(input_file_path).map(lambda l: l.split(","))
    header = small.first()
    if case_number==1:
        data = small.filter(lambda l: l!=header) \
                    .map(lambda u_b: (u_b[0], u_b[1]))
    if case_number==2:
        data = small.filter(lambda l: l!=header) \
                .map(lambda u_b: (u_b[1], u_b[0]))

    participants = data.groupByKey().map(lambda u_b: sorted(list(set(u_b[1]))))
    len_data = participants.count()

    candidates = participants.mapPartitions(lambda b: find_candidates(b, support, len_data)) \
                            .flatMap(lambda x: x).distinct() \
                            .sortBy(lambda x: (len(x), x))
    candidates = candidates.collect()    
    
    freq = participants.mapPartitions(lambda b: find_frequent(b, candidates)) \
                    .reduceByKey(add) \
                    .filter(lambda f: f[1]>=support).map(lambda f: f[0]) \
                    .sortBy(lambda f: (len(f), f))
    freq = freq.collect()

    
    with open(output_file_path, 'w+') as f:
        f.write('Candidates:\n' + output_form(candidates) + '\n\n' + 'Frequent Itemsets:\n' + output_form(freq))

    time_end = time.time()
    print('Duration:', time_end - time_start)
    
    