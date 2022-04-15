#!/usr/bin/python
# -*- coding: UTF-8 -*-

'''
    Task1: Min-Hash + LSH (3.5pts)

    Task description
    In this task, you will implement the Min-Hash and Locality Sensitive Hashing algorithms with Jaccard similarity to find similar business pairs in the train_review.json file. We focus on 0/1 ratings rather than the actual rating values in the reviews. In other words, if a user has rated a business, the user’s contribution in the characteristic matrix is 1; otherwise, the contribution is 0. Your task is to identify business pairs whose Jaccard similarity is >= 0.05.
'''


import sys
import json
import random
from collections import defaultdict
from itertools import combinations
from pyspark import SparkContext


def min_hash(rows, n_hash, p, m):
    hash_dict = defaultdict(list)
    for i in range(len(rows)):
        for j in range(n_hash):
            a, b = random.randint(1, 99999), random.randint(1, 99999)
            hash_dict[i].append(((a*i+b)%p)%m)
    return hash_dict


def LSH(signature, n_hash, n_bands):
    candidates = set()
    n_rows_per_band = n_hash//n_bands
    assert n_hash%n_bands==0, 'Wrong number of bands!'
    for b in range(n_bands):
        buckets = defaultdict(set)
        for sgnt in signature:
            start_row = b * n_rows_per_band
            vector = ()
            for r in range(n_rows_per_band):
                vector += (sgnt[1][start_row+r],)
            vector_hashed = hash(vector)
            buckets[vector_hashed].add(sgnt[0])
        for pairs in buckets.values():
            if len(pairs)>=2:
                for p in combinations(pairs, 2):
                    candidates.add(tuple(sorted(p)))
    return candidates



def main(params):
    
    input_file, output_file = params[1], params[2]

    sc = SparkContext.getOrCreate()
    review = sc.textFile(input_file).map(lambda r: json.loads(r)) \
                                    .map(lambda u_b: (u_b['user_id'], u_b['business_id']))
    users = review.map(lambda u_b: u_b[0]).distinct().collect()

    users_id = {}
    for i,u in enumerate(users):
        users_id[u] = i
    review_index = review.map(lambda u_b: (users_id[u_b[0]], u_b[1]))

    n_hash = 100
    users_hashed = min_hash(users, n_hash, p=9841, m=87319)

    signature = review_index.groupByKey().map(lambda u_bs: (u_bs[0], list(set(u_bs[1])))) \
                            .map(lambda uids_bs: (uids_bs[1], users_hashed[uids_bs[0]])) \
                            .map(lambda bs_uids: ((b, bs_uids[1]) for b in bs_uids[0])).flatMap(lambda b_uids: b_uids) \
                            .groupByKey().map(lambda b_uids: (b_uids[0], [list(uids) for uids in b_uids[1]])) \
                            .map(lambda b_us: (b_us[0], [min(uids) for uids in zip(*b_us[1])])).collect()

    candidates = LSH(signature, n_hash, n_bands=100)

    business_users = review.map(lambda u_b: (u_b[1], u_b[0])) \
                            .groupByKey().map(lambda b_u: (b_u[0], set(b_u[1]))).collectAsMap()

    
    with open(output_file, 'w+') as f:
        for p in candidates:
            b1, b2 = p[0], p[1]
            u1, u2 = business_users[b1], business_users[b2]
            sim = float(len(u1 & u2) / len(u1 | u2))
            if sim>=.05:
                f.write(json.dumps({'b1': b1, 'b2': b2, 'sim': sim}) + '\n')
            
            
if __name__ == '__main__':
    params = sys.argv
    main(params)
    
    