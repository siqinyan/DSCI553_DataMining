#!/usr/bin/python
# -*- coding: UTF-8 -*-


'''
    Task (12.5pts)
    
    Task description
    You will write the K-Means and Bradley-Fayyad-Reina (BFR) algorithms from scratch. You should implement K-Means as the in-memory clustering algorithm that you will use in BFR. You will iteratively load the data points from a file and process these data points with the BFR algorithm. See below pseudocode for your reference.
    
    for file in input_path:
        data_points = load(file)
        if first round:
            run K-means for initialization
        else:
            run BFR(data_points)
    In BFR, there are three sets of points that you need to keep track of: Discard set (DS), Compression set (CS), Retained set (RS). For each cluster in the DS and CS, the cluster is summarized by:
        - N: The number of points
        - SUM: the sum of the coordinates of the points
        - SUMSQ: the sum of squares of coordinates
'''


import sys
import time
import json
import math
import random
import copy
import csv
import os
import itertools
from collections import defaultdict
from pyspark import SparkContext




def calc_distance(A, B, std=None, distance_type='euclidean'):
    if distance_type=='euclidean':
        return float(math.sqrt(sum([(a-b)**2 for a,b in zip(A, B)])))
    elif distance_type=='mahalanobis':
        return float(math.sqrt(sum([((a-b)/sd)**2 for a,b,sd in zip(A, B, std)])))

def calc_col_avg(prev, cur, denominator=None):
    tmp_list = []
    tmp_list.append(prev)
    tmp_list.append(cur)
    if denominator is None:
        return [sum(i)/len(i) for i in zip(*tmp_list)]
    else:
        return [sum(i)/denominator for i in zip(*tmp_list)]

def export_file(data, output_file, output_type='json', export_type='w+'):
    if output_type=='json':
        with open(output_file, export_type) as f:
            f.writelines(json.dumps(data))
    elif output_type=='csv':
        with open(output_file, export_type, newline='') as f:
            writer = csv.writer(f)
            for k,v in data.items():
                writer.writerow(v)

                
                
class KMeans():
    def __init__(self, n_cluster, max_iter=300, seed=666):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.seed = seed
        self.centroid_info = {}
        self.centroid_stat_info = {}
        self.clusters_points = {}
        self.stable = {}
        self.data_dict = None

    
    def _check_data(self):
        if len(self.data_dict) < self.n_cluster:
            self.n_cluster = len(self.data_dict)
        
    def _init_centroids(self):
        random.seed(self.seed)
        for i,k in enumerate(random.sample(self.data_dict.keys(), self.n_cluster)):
            self.centroid_info.setdefault('c'+str(i), self.data_dict.get(k))
            self.centroid_stat_info.setdefault('c'+str(i), self.data_dict.get(k))
            self.clusters_points.setdefault('c'+str(i), list())
            self.stable.setdefault('c'+str(i), False)
    
    def _update_centroids(self):
        info_prev = copy.deepcopy(self.centroid_info)
        for c, ps in self.clusters_points.items():
            if not self.stable.get(c):
                tmp_list = []
                tmp_list.append(self.centroid_info.get(c))
                for p in ps:
                    tmp_list.append(self.data_dict.get(p))

                self.centroid_info[c] = [sum(i) / len(i) for i in zip(*tmp_list)]
                self.centroid_stat_info[c] = [sum([v**2 for v in i]) / len(i) for i in zip(*tmp_list)]

        return info_prev, self.centroid_info
        
        
    def fit(self, data_dict):
        self.data_dict = data_dict
        self._check_data()
        self._init_centroids()
        n_iter = 1
        while True: # can not be while n_iter<self.max_iter, will _clear()
            for k in self.data_dict:
                tmp_dict = {}
                for c in self.centroid_info:
                    tmp_dict[(c, k)] = calc_distance(self.centroid_info[c], self.data_dict[k])
                tmp_info = list(sorted(tmp_dict.items(), key=lambda kv: kv[1]))[:1]
                self.clusters_points[tmp_info[0][0][0]].append(tmp_info[0][0][1])

            info_prev, info_cur = self._update_centroids()
            if not self._updated(info_prev, info_cur) or n_iter>=self.max_iter:
                break
            self._clear()
            n_iter += 1

        return self.centroid_info, self.centroid_stat_info, self.clusters_points

    
    def _clear(self):
        for c in self.clusters_points:
            self.clusters_points[c] = []

    def _updated(self, prev, cur):
        for k in prev:
            value_prev = set(map(lambda v: round(v, 0), prev.get(k)))
            value_cur = set(map(lambda v: round(v, 0), cur.get(k)))
            if len(value_prev.difference(value_cur))==0:
                self.stable[k] = True
            else:
                self.stable[k] = False
                return True
        return False

    
    
    
class Cluster():
    def __init__(self):
        self.SUM_N = None
        self.SUMSQ_N = None
        self.clusters_points = None
        self.signature = None

    def init(self, info, stat, clusters_points):
        self.SUM_N = info
        self.SUMSQ_N = stat
        self.clusters_points = clusters_points
        self.n_point = 0
        self.n_dim = len(list(info.values())[0])
        self.STD = dict()
        self.calc_std()
        

    def calc_n_point(self):
        self.n_point = 0
        for _,v in self.clusters_points.items():
            if type(v)==list:
                self.n_point += len(v)
        return self.n_point
                
    def calc_n_cluster(self):
        return len(self.SUM_N)
        
    def calc_std(self):
        self.STD = {}
        for k in self.SUM_N:
            self.STD[k] = [math.sqrt(sq_n - sum_n**2) \
                               for sq_n, sum_n in zip(self.SUMSQ_N.get(k), self.SUM_N.get(k))]

        
    def update_centroid_info(self, tmp_clusters_points, tmp_data_dict):
        if len(tmp_clusters_points)>0:
            sum_n_prev = copy.deepcopy(self.SUM_N)
            sumsq_n_prev = copy.deepcopy(self.SUMSQ_N)
            clusters_points_prev = copy.deepcopy(self.clusters_points)
            for c, ps in tmp_clusters_points.items():
                tmp_list = []
                loc_prev1 = sum_n_prev.get(c)
                len_cluster_prev = len(clusters_points_prev.get(c))
                loc_prev1 = list(map(lambda v: v*len_cluster_prev, loc_prev1))
                for p in ps:
                    tmp_list.append(tmp_data_dict.get(p))

                loc_prev2 = [sum(i) for i in zip(*tmp_list)]
                len_cluster_new = len(tmp_list) + len_cluster_prev
                self.SUM_N[c] = calc_col_avg(loc_prev1, loc_prev2, denominator=len_cluster_new)

                loc_prev3 = sumsq_n_prev.get(c)
                loc_prev3 = list(map(lambda v: v*len_cluster_prev, loc_prev3))
                loc_prev4 = [sum([v**2 for v in i]) for i in zip(*tmp_list)]
                self.SUMSQ_N[c] = calc_col_avg(loc_prev3, loc_prev4, denominator=len_cluster_new)

            self.calc_std()
            self.update_clusters_points(tmp_clusters_points)

    def update_clusters_points(self, tmp_clusters_points):
        if len(tmp_clusters_points)>0:
            combined = defaultdict(list)
            for k, v in itertools.chain(self.clusters_points.items(), tmp_clusters_points.items()):
                combined[k] += v
                
            self.clusters_points = combined
            
            
    def get_points_of_cluster(self, k):
        return list(self.clusters_points.get(k))
    
    def get_centroid_of_clustuer(self, k):
        return list(self.SUM_N.get(k))
    
    def get_std_of_clustuer(self, k):
        return list(self.STD.get(k))
    
    def get_sumsq_of_clustuer(self, k):
        return list(self.SUMSQ_N.get(k))
    

class DS(Cluster):
    def __init__(self):
        Cluster.__init__(self)
        self.signature = 'DS'

    def merge(self, ds_clusters, cs_cluster_sumsq, cs_cluster_centroid, cs_clusters_points):
        loc_prev1 = self.get_centroid_of_clustuer(ds_clusters)
        len_prev = len(self.get_points_of_cluster(ds_clusters))
        loc_prev1 = list(map(lambda v: v*len_prev, loc_prev1))

        len_cur = len(cs_clusters_points)
        loc_prev2 = list(map(lambda v: v*len_cur, cs_cluster_centroid))
        loc_cur = calc_col_avg(loc_prev1,
                               loc_prev2,
                               denominator=len_prev+len_cur)

        sumsq_prev1 = list(map(lambda v: v*len_prev, self.get_sumsq_of_clustuer(ds_clusters)))
        sumsq_prev2 = list(map(lambda v: v*len_cur, cs_cluster_sumsq))
        sumsq_cur = calc_col_avg(sumsq_prev1,
                                 sumsq_prev2,
                                 denominator=len_prev+len_cur)

        self.SUM_N.update({ds_clusters: loc_cur})
        self.clusters_points[ds_clusters].extend(cs_clusters_points)
        self.SUMSQ_N.update({ds_clusters: sumsq_cur})

        self.calc_n_point()
        self.calc_std()


class CS(Cluster):
    def __init__(self):
        Cluster.__init__(self)
        self.signature = 'CS'
        self.r2c_index = 0
        self.merge_index = 0

    def remove_cluster(self, c):
        self.SUM_N.pop(c)
        self.SUMSQ_N.pop(c)
        self.clusters_points.pop(c)
        self.STD.pop(c)
        self.calc_n_point()

    def delta_update(self, info, stat, clusters_points):
        if len(info)!=0:
            for k in info:
                self.SUM_N.update({'r2c'+ str(self.r2c_index): info.get(k)})
                self.SUMSQ_N.update({'r2c'+ str(self.r2c_index): stat.get(k)})
                self.clusters_points.update({'r2c'+ str(self.r2c_index): clusters_points.get(k)})
                self.calc_std()
                self.r2c_index += 1

    def merge(self, cluster1, cluster2):
        loc_cur = calc_col_avg(list(self.SUM_N[cluster1]), list(self.SUM_N[cluster2]))
        sumsq_cur = calc_col_avg(list(self.SUMSQ_N[cluster1]), list(self.SUMSQ_N[cluster2]))
        clusters_points_cur = list(self.clusters_points[cluster1])
        clusters_points_cur.extend(list(self.clusters_points[cluster2]))

        cluster_cur = 'm'+str(self.merge_index)
        self.SUM_N.pop(cluster1)
        self.SUM_N.pop(cluster2)
        self.SUM_N.update({cluster_cur: loc_cur})

        self.SUMSQ_N.pop(cluster1)
        self.SUMSQ_N.pop(cluster2)
        self.SUMSQ_N.update({cluster_cur: sumsq_cur})

        self.clusters_points.pop(cluster1)
        self.clusters_points.pop(cluster2)
        self.clusters_points.update({cluster_cur: clusters_points_cur})

        self.calc_std()
        self.merge_index += 1

    def sort(self):
        result = defaultdict(list)
        for c in self.clusters_points:
            result[c] = sorted(self.clusters_points[c])

        return result


class RS():
    def __init__(self):
        self.remaining_set = {}

    def add(self, data):
        self.remaining_set.update(data)
        
    def gather(self, left):
        self.remaining_set = left
        
    def calc_n_point(self):
        return len(self.remaining_set)

    @classmethod
    def getSignature(cls):
        return 'RS'

    
class IntermediateRecords():
    def __init__(self):
        self.intermediate_records = {}
        self.intermediate_records["header"] = ("round_id", 
                                               "nof_cluster_discard", "nof_point_discard",
                                               "nof_cluster_compression", "nof_point_compression",
                                               "nof_point_retained")

    def save_check_point(self, round_id, DS, CS, RS):
        self.intermediate_records[round_id] = (round_id,
                                               DS.calc_n_cluster(), DS.calc_n_point(),
                                               CS.calc_n_cluster(), CS.calc_n_point(),
                                               RS.calc_n_point())
#         print("{} -> DS_INFO: C:{} NUM:{} | CS_INFO: C:{} NUM:{} | RS_INFO: NUM:{}".format(
#             round_id, DS.calc_n_cluster(), DS.calc_n_point(), CS.calc_n_cluster(), CS.calc_n_point(), RS.calc_n_point()
#         ))

    def export(self, output_file):
        export_file(self.intermediate_records, output_file, output_type='csv')

       
    
        
def check_clustering(info, stat, clusters_points):
    remaining_points = {}
    tmp_clusters_points = copy.deepcopy(clusters_points)
    for c, ps in tmp_clusters_points.items():
        if len(ps)<=1:
            if len(ps)!=0:
                remaining_points.update({ps[0]: info.get(c)})
            clusters_points.pop(c)
            info.pop(c)
            stat.pop(c)

    return info, stat, clusters_points, remaining_points


def assign_set(data, alpha, DS=None, CS=None, cluster_type=''):
    if DS and cluster_type==DS.signature:
        ds_dim = DS.n_dim
        distance = float('inf')
        key_to_ds = None
        for k, loc in DS.SUM_N.items():
            tmp_distance = calc_distance(data[1], loc, DS.STD.get(k), distance_type='mahalanobis')
            if tmp_distance < alpha *math.sqrt(ds_dim) and tmp_distance < distance:
                distance = tmp_distance
                key_to_ds = (k, data[0])

        if key_to_ds:
            yield tuple((key_to_ds, data[1], False))
        else:
            yield tuple((("-1", data[0]), data[1], True))

    elif CS and cluster_type==CS.signature:
        cs_dim = CS.n_dim
        distance = float('inf')
        key_to_cs = None
        for k, loc in CS.SUM_N.items():
            tmp_distance = calc_distance(data[1], loc, CS.STD.get(k), distance_type='mahalanobis')
            if tmp_distance < alpha *math.sqrt(cs_dim) and tmp_distance < distance:
                distance = tmp_distance
                key_to_cs = (k, data[0])

        if key_to_cs:
            yield tuple((key_to_cs, data[1], False))
        else:
            yield tuple((("-1", data[0]), data[1], True))


def merge_CS(alpha, CS):
    cs_dim = CS.n_dim
    CS_prev = copy.deepcopy(CS)
    set_avail = set(list(CS_prev.SUM_N.keys()))
    for pair in itertools.combinations(list(CS_prev.SUM_N.keys()), 2):
        if pair[0] in set_avail and pair[1] in set_avail:
            distance = calc_distance(CS_prev.get_centroid_of_clustuer(pair[0]),
                                     CS_prev.get_centroid_of_clustuer(pair[1]),
                                     std=CS_prev.get_std_of_clustuer(pair[0]),
                                     distance_type='mahalanobis')
            if distance < alpha *math.sqrt(cs_dim):
                CS.merge(pair[0], pair[1])
                set_avail.discard(pair[0])
                set_avail.discard(pair[1])


def assign_CS2DS(alpha, DS, CS):
    ds_dim = DS.n_dim
    DS_prev = copy.deepcopy(DS)
    CS_prev = copy.deepcopy(CS)
    for centroid_cs in CS_prev.SUM_N:
        for centroid_ds in DS_prev.SUM_N:
            distance = calc_distance(CS_prev.get_centroid_of_clustuer(centroid_cs),
                                     DS_prev.get_centroid_of_clustuer(centroid_ds),
                                     DS_prev.get_std_of_clustuer(centroid_ds),
                                     distance_type='mahalanobis')
            if distance < alpha *math.sqrt(ds_dim):
                DS.merge(centroid_ds,
                         CS_prev.get_sumsq_of_clustuer(centroid_cs),
                         CS_prev.get_centroid_of_clustuer(centroid_cs),
                         CS_prev.get_points_of_cluster(centroid_cs))
                CS.remove_cluster(centroid_cs)
                break

                
def export_clusters_points(DS, CS, RS, output_file):
    result = defaultdict()
    for c in DS.clusters_points:
        [result.setdefault(str(p), int(c[1:])) for p in DS.get_points_of_cluster(c)]
        
    for c in CS.clusters_points:
        [result.setdefault(str(p), -1) for p in CS.get_points_of_cluster(c)]

    for k in RS.remaining_set:
        result.setdefault(str(k), -1)

    result_sorted = dict(sorted(result.items(), key=lambda kv: int(kv[0])))
    export_file(result_sorted, output_file, output_type='json')
    
    
    
if __name__ == '__main__':
    time_start = time.time()

    params = sys.argv
    input_path, n_cluster = params[1], int(params[2])
    out_file1, out_file2 = params[3], params[4]

    sc = SparkContext.getOrCreate()
    sc.setLogLevel("WARN")

    alpha = 3
    n_times = 3
    discard_set = DS()
    compression_set = CS()
    retained_set = RS()
    intermediate_records = IntermediateRecords()

    
    for i, fp in enumerate(sorted(os.listdir(input_path))):
        path = input_path + "/" + fp
        data = sc.textFile(path).map(lambda r: r.split(",")) \
                                .map(lambda kvs: (int(kvs[0]), list(map(eval, kvs[1:]))))

        if i==0:
            n_data = data.count()
            n_sample = 10000 if n_data > 10000 else int(n_data *.1)
            data_sample = data.filter(lambda kv: kv[0]<n_sample).collectAsMap()

            ds_info, ds_stat, ds_clusters_points = KMeans(n_cluster, max_iter=5).fit(data_sample)

            discard_set.init(ds_info, ds_stat, ds_clusters_points)

            data_rest = data.filter(lambda kv: kv[0]>=n_sample).collectAsMap()
            info, stat, clusters_points = KMeans(n_cluster *n_times, max_iter=3).fit(data_rest)

            cs_info, cs_stat, cs_clusters_points, remaining = check_clustering(info, stat, clusters_points)
            compression_set.init(cs_info, cs_stat, cs_clusters_points)
            retained_set.add(remaining)

        else:
            data_sets1 = data.flatMap(lambda point: assign_set(point, alpha, DS=discard_set, cluster_type='DS'))

            tmp_ds = data_sets1.filter(lambda flag: flag[2] is False) \
                                .map(lambda flag: (flag[0], flag[1]))

            tmp_ds_cluster_points = tmp_ds.map(lambda k: k[0]) \
                                          .groupByKey().mapValues(list).collectAsMap()

            tmp_ds_data_dict = tmp_ds.map(lambda k: (k[0][1], list(k[1]))).collectAsMap()
            discard_set.update_centroid_info(tmp_ds_cluster_points, tmp_ds_data_dict)


            data_sets2 = data_sets1.filter(lambda flag: flag[2] is True) \
                                    .map(lambda flag: (flag[0][1], flag[1])) \
                                    .flatMap(lambda point: assign_set(point, alpha, CS=compression_set, cluster_type='CS'))

            tmp_cs = data_sets2.filter(lambda flag: flag[2] is False).map(lambda flag: (flag[0], flag[1]))

            tmp_cs_cluster_points = tmp_cs.map(lambda k: k[0]) \
                                          .groupByKey().mapValues(list).collectAsMap()
            tmp_cs_data_dict = tmp_cs.map(lambda k: (k[0][1], list(k[1]))).collectAsMap()
            compression_set.update_centroid_info(tmp_cs_cluster_points, tmp_cs_data_dict)

            remaining_data_dict = data_sets2.filter(lambda flag: flag[2] is True) \
                                            .map(lambda flag: (flag[0][1], flag[1])).collectAsMap()
            retained_set.add(remaining_data_dict)

            info, stat, clusters_points = KMeans(n_cluster *n_times, max_iter=5).fit(retained_set.remaining_set)
            cs_info, cs_stat, cs_clusters_points, remaining2 = check_clustering(info, stat, clusters_points)
            compression_set.delta_update(cs_info, cs_stat, cs_clusters_points)
            retained_set.gather(remaining2)

            merge_CS(alpha, compression_set)

        if i+1==len(os.listdir(input_path)):
            assign_CS2DS(alpha, discard_set, compression_set)

        intermediate_records.save_check_point(i+1, discard_set, compression_set, retained_set)

        
    intermediate_records.export(out_file2)
    export_clusters_points(discard_set, compression_set, retained_set, out_file1)
    print("Duration: %d s." % (time.time() - time_start))
    
    