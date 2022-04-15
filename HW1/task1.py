#!/usr/bin/python
# -*- coding: UTF-8 -*-

'''
    Task1: Data Exploration (4.5pts)
    Task description: You will explore the review dataset and write a program to answer the following questions.
'''

import sys
import json
from datetime import datetime
from pyspark import SparkConf, SparkContext
from operator import add


def main(params):
    input_file, output_file, stopwords = params[1], params[2], params[3], 
    y, m, n = int(params[4]), int(params[5]), int(params[6])

    results = {}

#     conf = SparkConf()
#     sc = SparkContext(conf=conf)
    sc = SparkContext.getOrCreate()
#     sc.setLogLevel("OFF")
    review = sc.textFile(input_file).map(lambda r: json.loads(r))

    ### A. The total number of reviews (0.5pts)
    results['A'] = review.map(lambda r: r['review_id']).count()

    ### B. The number of reviews in a given year, y (1pts)
    results['B'] = review.map(lambda r: r['date']) \
                        .filter(lambda r: datetime.strptime(r, '%Y-%m-%d %H:%M:%S').year==y).count()

    ### C. The number of distinct users who have written the reviews (1pts)
    results['C'] = review.map(lambda r: r['user_id']).distinct().count()

    ### D. Top m users who have the largest number of reviews and its count (1pts)
    results['D'] = review.map(lambda r: (r['business_id'], r['user_id'])).map(lambda r: (r[1], 1)) \
                         .reduceByKey(add).top(m, key=lambda r: r[1])

    ### E. Top n frequent words in the review text. The words should be in lower cases. The following punctuations “(”, “[”, “,”, “.”, “!”, “?”, “:”, “;”, “]”, “)” and the given stopwords are excluded (1pts)
    def exclude_words(word):
        if word not in stopwords:
            re = ''
            for w in word:
                if w not in punctuations:
                    re += w
            return re

    punctuations = set(['(', '[', ',', '.', '!', '?', ':', ';', ']', ')'])
    stopwords = set(w.strip() for w in open(stopwords))

    tmp = review.map(lambda r: r['text']).flatMap(lambda t: t.lower().split(' ')) \
                .map(lambda w: (exclude_words(w), 1)).filter(lambda w: w[0]!=None).reduceByKey(add) \
                .sortBy(lambda c: (-c[1], c[0])).take(n)
    results['E'] = [w for w,_ in tmp]


    with open(output_file, 'w+') as o:
        json.dump(results, o)
        

        
if __name__ == '__main__':
    params = sys.argv
    main(params)
    
