#!/usr/bin/python
# -*- coding: UTF-8 -*-

'''
    Task2: Content-based Recommendation System (4pts)

    Task description
    In this task, you will build a content-based recommendation system by generating profiles from review texts for users and businesses in the train_review.json file. Then you will use the model to predict if a user prefers to review a given business by computing the cosine similarity between the user and item profile vectors.
    During the training process, you will construct the business and user profiles as follows.
'''


import sys
import re
import json
import math
from collections import defaultdict
from pyspark import SparkContext, SparkConf
from operator import add



def concat_words(text):
    text = re.sub(r'[^\w\s]', ' ', text)
    text = text.split()
    return text
    
    
def parse(text, stopwords):
    text = re.sub(r'[^\w\s]', ' ', text)
    text = ''.join(c for c in text if not c.isdigit())
    words = text.split()
#     text = [w for w in words if w not in stopwords and count_words[w]>=.000001*n_words]
    text = [w for w in words if w not in stopwords]
    return text


def calc_tf(text):
    words_count = defaultdict(int)
    for w in text:
        words_count[w] += 1
    max_tf = max(words_count.values())
    tf = {}
    for w,c in words_count.items():
        tf[w] = c / max_tf
    return tf


def func_sort(t_s):
    t_s.sort(key=lambda x: -x[1])
    return t_s

    
    
def main(params):
    
    train_file, model_file, stopwords = params[1], params[2], params[3]

#     conf = SparkConf()
#     conf.set("spark.executor.memory", "4g")
#     conf.set("spark.driver.memory", "4g")
#     sc = SparkContext.getOrCreate(conf)
    sc = SparkContext.getOrCreate()

    review = sc.textFile(train_file).map(lambda r: json.loads(r)) \
                .map(lambda r: (r['user_id'], r['business_id'], r['text']))

    stopwords = set(w.strip() for w in open(stopwords))

    ### Concatenating all reviews for a business as one document and parsing the document, such as removing the punctuations, numbers, and stopwords. Also, you can remove extremely rare words to reduce the vocabulary size. Rare words could be the ones whose frequency is less than 0.0001% of the total number of words.
#     text_concat = review.map(lambda u_b_t: u_b_t[2].lower()).flatMap(lambda t: concat_words(t))
#     n_words = text_concat.count()
#     count_words = text_concat.map(lambda t: (t, 1)).reduceByKey(add).collect()
#     count_words = dict(count_words)

    business_text = review.repartition(50).map(lambda u_b_t: (u_b_t[1], u_b_t[2].lower())) \
                        .map(lambda b_t: (b_t[0], parse(b_t[1], stopwords))) \
                        .groupByKey().map(lambda b_t: (b_t[0], [w for grp in b_t[1] for w in grp]))

    ### Measuring word importance using TF-IDF, i.e., term frequency multiply inverse doc frequency.
    term_frequency = business_text.mapValues(lambda b_t: calc_tf(b_t)) \
                                .flatMap(lambda b_tf: [(b_tf[0], (t, f)) for t,f in b_tf[1].items()]) \
                                .map(lambda b_t_f:(b_t_f[1][0], (b_t_f[0], b_t_f[1][1])))

    n_business = review.map(lambda u_b_t: u_b_t[1]).distinct().count()
    inverse_document_frequency = business_text.flatMap(lambda b_t: [(w, b_t[0]) for w in b_t[1]]) \
                                            .groupByKey().mapValues(lambda t: math.log(n_business/len(set(t)), 2))

    ### Using top 200 words with the highest TF-IDF scores to describe the document
    profile = term_frequency.join(inverse_document_frequency) \
                    .map(lambda t_bf_idf: (t_bf_idf[1][0][0], (t_bf_idf[0], t_bf_idf[1][0][1]*t_bf_idf[1][1]))) \
                    .groupByKey().mapValues(lambda t_s: [x for x in func_sort(list(t_s))[:200]])

    ### Creating a Boolean vector with these significant words as the business profile
    words = profile.flatMap(lambda b_t_s: [t_s[0] for t_s in b_t_s[1]]).distinct().collect()
    id_words = {}
    for i,w in enumerate(words):
        id_words[w] = i
    profile = profile.map(lambda b_t_s:(b_t_s[0], [(id_words[t_s[0]], t_s[1]) for t_s in b_t_s[1]])).collect()

    profile_dict = {}
    for b_tid_s in profile:
        profile_dict[b_tid_s[0]] = b_tid_s[1]

    ### Creating a Boolean vector for representing the user profile by aggregating the profiles of the items that the user has reviewed
    profile_user = review.map(lambda u_b_t: (u_b_t[0], u_b_t[1])) \
                        .groupByKey().map(lambda u_b: (u_b[0], [b for b in set(u_b[1])])) \
                        .map(lambda u_bs: (u_bs[0], [profile_dict[b] for b in u_bs[1]])) \
                        .map(lambda u_b_t_s: (u_b_t_s[0], list(set([t_s for b_t_s in u_b_t_s[1] for t_s in b_t_s])))) \
                        .map(lambda u_t_s: (u_t_s[0], [t_s[0] for t_s in func_sort(u_t_s[1])[:500]])).collect()

    with open(model_file, 'w+') as f:
        for b_t_s in profile:
            f.write(json.dumps({'business': b_t_s[0], 'features': [t_s[0] for t_s in b_t_s[1]]}) + '\n')
        for u_t_s in profile_user:
            f.write(json.dumps({'user': u_t_s[0], 'features': u_t_s[1]}) + '\n')

        
            
if __name__ == '__main__':
    params = sys.argv
    main(params)
    
    