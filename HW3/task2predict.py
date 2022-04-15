#!/usr/bin/python
# -*- coding: UTF-8 -*-

'''
    Task2: Content-based Recommendation System (4pts)

    Task description
    During the prediction process, you will estimate if a user would prefer to review a business by computing the cosine distance between the profile vectors. The (user, business) pair is valid if their cosine similarity is >= 0.01. You should only output these valid pairs.
'''


import sys
import json
import math
from pyspark import SparkContext, SparkConf


def cos_sim(profile_user, profile_business):
    sim = 0
    if profile_user and profile_business:
        u, b = set(profile_user), set(profile_business)
        sim = len(u & b) / (math.sqrt(len(u)) * math.sqrt(len(b)))
    return sim
    
    
def main(params):
    
    test_file, model_file, output_file = params[1], params[2], params[3]

    sc = SparkContext.getOrCreate()

    review = sc.textFile(test_file).map(lambda r: json.loads(r)) \
                                .map(lambda r: (r['user_id'], r['business_id']))

    model = sc.textFile(model_file).map(lambda r: json.loads(r))

    profile = model.filter(lambda r: 'business' in r) \
                    .map(lambda r: (r['business'], r['features'])).collect()
    profile_business = {}
    for b_f in profile:
        profile_business[b_f[0]] = b_f[1]

    profile = model.filter(lambda r: 'user' in r) \
                    .map(lambda r: (r['user'], r['features'])).collect()
    profile_user = {}
    for u_f in profile:
        profile_user[u_f[0]] = u_f[1]

    pred = review.map(lambda u_b: (u_b, cos_sim(profile_user.get(u_b[0]), profile_business.get(u_b[1])))) \
                .filter(lambda u_b_s: u_b_s[1]>=.01).collect()

    with open(output_file, 'w+') as f:
        for u_b_s in pred:
            f.write(json.dumps({'user_id': u_b_s[0][0], 'business_id': u_b_s[0][1], 'sim': u_b_s[1]}) + '\n')
        
            
if __name__ == '__main__':
    params = sys.argv
    main(params)
    
    