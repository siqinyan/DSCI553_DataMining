#!/usr/bin/python
# -*- coding: UTF-8 -*-

'''
    Task2: Exploration on Multiple Datasets (4pts)
    Task description: In task2, you will explore the two datasets together (i.e., review and business) and write a program to compute the average stars for each business category and output top n categories with the highest average stars. The business categories should be extracted from the “categories” tag in the business file. The categories should be split by comma and removed leading and trailing spaces. No other operations needed to process contents in the “categories” tag in the business file. Stars are extracted from the review dataset. Two datasets are joined by “business_id” tag. You need to implement a version without Spark (2pts) and a version with Spark (2pts). You could then compare their performance yourself (not graded).
'''

import sys
import json
from pyspark import SparkConf, SparkContext


def main(params):
    review_file, business_file, output_file = params[1], params[2], params[3], 
    if_spark, n = params[4], int(params[5])
    
    results = {}

    ##### W/ Spark
    if if_spark=='spark':
#         conf = SparkConf()
#         sc = SparkContext(conf=conf)
        sc = SparkContext.getOrCreate()

        review = sc.textFile(review_file).map(lambda r: json.loads(r))
        business = sc.textFile(business_file).map(lambda r: json.loads(r))

        review_rdd = review.map(lambda r: (r['business_id'], r['stars']))
        business_rdd = business.map(lambda b: (b['business_id'], b['categories']))

        stars = review_rdd.groupByKey().mapValues(list).map(lambda bid: (bid[0], (sum(bid[1]), len(bid[1]))))
        categories = business_rdd.filter(lambda ctg: ctg[1]!=None) \
                                .mapValues(lambda ctgs: [ctg.strip() for ctg in ctgs.split(',')])
        aggregated = categories.leftOuterJoin(stars)

        tmp = aggregated.map(lambda id_c_s: id_c_s[1]).filter(lambda c_s: c_s[1]!=None) \
                        .flatMap(lambda c_s: [(c, c_s[1]) for c in c_s[0]]) \
                        .reduceByKey(lambda x,y: (x[0]+y[0], x[1]+y[1])).mapValues(lambda s: s[0]/s[1]) \
                        .sortBy(lambda r: (-r[1], r[0])).take(n)
        results['result'] = [[c_s[0], c_s[1]] for c_s in tmp]


    ##### W/O Spark
    else:
        def json_to_dict(file, keys):
            result = []
            data = open(file, encoding="utf8").readlines()
            for d in data:
                dictionary = {}
                for k in keys:
                    dictionary[k] = json.loads(d)[k]
                result.append(dictionary)

            return result

        stars = json_to_dict(review_file, keys=['business_id', 'stars'])
        categories = json_to_dict(business_file, keys=['business_id', 'categories'])

        stars_agg = {}
        for data in stars:
            bs_id, s = data['business_id'], data['stars']
            if bs_id not in stars_agg:
                stars_agg[bs_id] = [s, 1]
            else:
                stars_agg[bs_id][0] += s
                stars_agg[bs_id][1] += 1

        ctgr_agg = {}
        for data in categories:
            bs_id, ctgrs = data['business_id'], data['categories']
            if ctgrs!=None:
                ctgr_agg[bs_id] = [ctgr.strip() for ctgr in ctgrs.split(',')]

#         joined_dict = {} #{ctgr: (sum, len)}
#         for bs_id, ctgrs in ctgr_agg.items():
#             for ctgr in ctgrs:
#                 if bs_id in stars_agg:
#                     if ctgr not in joined_dict:
#                         joined_dict[ctgr] = stars_agg[bs_id]
#                     else:
#                         joined_dict[ctgr][0] += stars_agg[bs_id][0]
#                         joined_dict[ctgr][1] += stars_agg[bs_id][1]
        joined_dict = {}
        for bs_id, s in stars_agg.items():
            if bs_id in ctgr_agg and ctgr_agg[bs_id]!=None:
                for c in ctgr_agg[bs_id]:
                    if c not in joined_dict:
                        joined_dict[c] = s
                    else:
        #                 joined_dict[c][0] += s[0]
        #                 joined_dict[c][1] += s[1]
                        s1 = joined_dict[c][0] + s[0]
                        s2 = joined_dict[c][1] + s[1]
                        joined_dict[c] = [s1, s2]

        results_dict = {k: v[0]/v[1] for k,v in joined_dict.items()}
        results_dict = dict(sorted(results_dict.items(), key=lambda item: (-item[1], item[0]))[:n])
        results['result'] = [[c, s] for c,s in results_dict.items()]


    with open(output_file, 'w+') as o:
        json.dump(results, o)
        

        
if __name__ == '__main__':
    params = sys.argv
    main(params)
    
