#!/usr/bin/python
# -*- coding: UTF-8 -*-

'''
    Task 2: Real-world data set -Yelp data- (4.0 pts)
    In task2, you will explore the Yelp dataset to find the frequent business sets (only case 1). You will jointly
    use the business.json and review.json to generate the input user-business CSV file yourselves.
    
    (1) Data preprocessing
    You need to generate a sample dataset from business.json and review.json with the following steps:
    1. The state of the business you need is Nevada, i.e., filtering ‘state’== ‘NV’.
    2. Select “user_id” and “business_id” from review.json whose “business_id” is from Nevada. Each line in the CSV file would be “user_id1, business_id1”.
    3. The header of CSV file should be “user_id,business_id”
    You need to save the dataset in CSV format.
'''


import csv
import json
from pyspark import SparkContext


if __name__ == '__main__':
    
    review_path = './HW1/review.json'
    business_path = './HW1/business.json'
    state = 'NV'
    output_path = './HW2/user_business.csv'

    sc = SparkContext.getOrCreate()

    review = sc.textFile(review_path).map(lambda r: json.loads(r)) 
    business = sc.textFile(business_path).map(lambda r: json.loads(r))

    business_need = business.map(lambda r: (r['business_id'], r['state'])) \
                            .filter(lambda r: r[1]==state).map(lambda r: r[0])
    business_need = business_need.collect()

    user_business = review.map(lambda r: (r['user_id'], r['business_id'])) \
                   .filter(lambda r: r[1] in business_need)
    user_business = output.collect()

    with open(output_path, 'w+') as f:
        writer = csv.writer(f)
        writer.writerow(['user_id', 'business_id'])
        for r in user_business:
            writer.writerow(r)
        
