#!/usr/bin/python
# -*- coding: UTF-8 -*-

'''
    Task3: Partition (4pts)
    Task description: In this task, you will learn how partitions work in the RDD. You need to compute the businesses that have more than n reviews in the review file. Other than the default way of partitioning the RDD, you should also design a customized partition function to improve computational efficiency. The “partition_type” is a hyperparameter in your program to decide which partition method to use. For either the default or the customized partition function, you need to show the number of partitions for the RDD, the number of items per partition, and the businesses that have more than n reviews (1pts for each partition type). Your customized partition function should improve the computational efficiency, i.e., reducing the time duration of execution (2pts).
'''

import sys
import json
from pyspark import SparkConf, SparkContext
from operator import add

    
def main(params):
    input_file, output_file = params[1], params[2]
    partition_type, n_partitions, n = params[3], int(params[4]), int(params[5])

    results = {}

#     conf = SparkConf()
#     sc = SparkContext(conf=conf)
    sc = SparkContext.getOrCreate()

    review = sc.textFile(input_file).map(lambda r: json.loads(r))

    bs_id_rdd = review.map(lambda r: (r['business_id'], 1))

    if partition_type!='default':
            bs_id_rdd = bs_id_rdd.partitionBy(n_partitions, lambda bs_id: ord(bs_id[0])+ord(bs_id[-1]))

    results['n_partitions'] = bs_id_rdd.getNumPartitions()
    results['n_items'] = bs_id_rdd.glom().map(len).collect()
    tmp = bs_id_rdd.reduceByKey(add).filter(lambda c: c[1]>n).collect()
    results['result'] = [[bs_id, c] for bs_id, c in tmp]

    
    with open(output_file, 'w+') as o:
        json.dump(results, o)

        
if __name__ == '__main__':
    params = sys.argv
    main(params)
    
