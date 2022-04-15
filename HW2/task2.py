#!/usr/bin/python
# -*- coding: UTF-8 -*-

'''
    Task 2: Real-world data set -Yelp data- (4.0 pts)
    In task2, you will explore the Yelp dataset to find the frequent business sets (only case 1). You will jointly
    use the business.json and review.json to generate the input user-business CSV file yourselves.
    
    (2) Apply SON algorithm
    The requirements for task 2 are similar to task 1. However, you will test your implementation with the large dataset you just generated. For this purpose, you need to report the total execution time. For this execution time, we take into account also the time from reading the file till writing the results to the output file. You are asked to find the frequent business sets (only case 1) from the file you just generated. The following are the steps you need to do:
    1. Reading the user_business CSV file in to RDD and then build the case 1 market-basket model; 
    2. Find out qualified users who reviewed more than k businesses. (k is the filter threshold);
    3. Apply the SON algorithm code to the filtered market-basket model;
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
    filter_threshold, support = int(params[1]), int(params[2])
    input_file_path, output_file_path = params[3], params[4]

    sc = SparkContext.getOrCreate()
    user_business = sc.textFile(input_file_path).map(lambda l: l.split(","))
    header = user_business.first()
    data = user_business.filter(lambda l: l!=header) \
                        .map(lambda u_b: (u_b[0], u_b[1]))

    participants = data.groupByKey().map(lambda u_b: sorted(list(set(u_b[1])))) \
                        .filter(lambda b: len(b)>filter_threshold)
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
    
    