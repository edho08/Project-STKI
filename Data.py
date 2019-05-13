# -*- coding: utf-8 -*-
from functools import reduce as rd
import numpy as np
from itertools import groupby, product
from operator import itemgetter
class Dataset:
    """
    Class ini adalah library untuk mengolah data
    """
    def __init__(self, arr):
        self.arr = arr
    def map(self, _function):
        return Dataset([_function(data) for data in self.arr])
    def flatMap(self, _function):
        return Dataset(flatten([_function(data) for data in flatten(self.arr)])) 
    def sortBy(self, _function):
        return Dataset(sorted(self.arr, key=_function))
    def reduce(self, _function):
        return rd(_function, self.arr)
    def reduceByKey(self, _function):
        inter = lambda x, y : (x[0], _function(x[1:],y[1:]) if len(x)>2 and len(y)>2 else _function(x[1],y[1]))
        return Dataset([rd(inter, group) for _, group in groupby(sorted(self.arr, key=lambda x:x[0]), key=itemgetter(0))])
    def collect(self):
        return self.arr
    def cartesian(self,dataset):
        cartes = [self.arr, dataset.arr]
        return Dataset([(element[0],element[1]) for element in product(*cartes)])
    def filter(self, _function):
        return Dataset(list(filter(_function, self.arr)))
    def join(self, dataset):
        return self.cartesian(dataset).filter(lambda x : x[0][0] == x[1][0]).map(lambda x : (x[0][0], [x[0][1:], x[1][1:]]))
    def leftOuterJoin(self, dataset):
        d1 = self.arr
        d2 = dataset.arr
        dicti = {}
        ret = []
        for d in d2:
            if d[0] in dicti:
                dicti[d[0]].append(d[1:])
            else:
                dicti[d[0]] = [d[1:]]
        for d in d1:
            if not d[0] in dicti:
                ret.append((d[0], ((d[1:]),[])))
            else:
                for r in dicti[d[0]]:
                    ret.append((d[0], ((d[1:]),(r))))

        return Dataset(ret)
    '''
        def leftOuterJoin(self, dataset):
        d1 = self.arr
        d2 = dataset.arr
        d3 = []
        for left in d1:
            isHaveMatch = False
            for right in d2:
                if left[0] == right[0]:
                    isHaveMatch = True
                    d3.append((left[0], (left[1:],right[1:])))
                    break
            if not isHaveMatch :
                d3.append((left[0], (left[1:],[])))
            
        return Dataset(d3)
    '''
    def append(self, dataset):
        return Dataset(self.arr + dataset.arr)
    def count(self):
        return len(self.arr)
    def unique(self):
        return Dataset(list(set(self.arr)))
    def foreach(self, _function):
        for element in self.arr: _function(element)
        
        
def flatten(arr):
    temp = []
    if isinstance(arr,str):
        return arr
    for i in range(len(arr)):
        if isinstance(arr[i],list) and not isinstance(arr[i],str):
            temp = temp + flatten(arr[i])
        else:
            temp.append(arr[i])
    return temp


'''
Db1 = Dataset([[1,2,[1,2], 5], [2,3]])
Db2 = Dataset([[1,4, 7, 9],[2,2]])
Db3 = Dataset(["jika aku ingin menjadi aku ingin menjadi aku ingin menjadi", "aku jika menjadi ingin aku", ["aku ingin","jika aku"]])
DB4 = Dataset([["jika", "ji"], ["aku", "a"]])
AD = Db3.flatMap(lambda x : x.strip().split(" ")).map(lambda x : (x, 1))
dcount = Db3.flatMap(lambda x : x.strip().split()).map(lambda x : (x, 1))
d = dcount.leftOuterJoin(DB4).map(lambda x : (x[1][1][0] if len(x[1][1])>0 else x[0]))
print(dcount.reduceByKey(lambda x,y: x+y).collect())


d1={"foo": 3, "baz": -1, "bar": 5}
d2={"foo": 3, "ven": 10, "bar": 5}
d_1_2= dict(list(d1.items()) + list(d2.items()))
print(d_1_2)
print(dict([[k, d2.get(k, 0)] for k in d_1_2] ))#right outer join
dict([[k, d1.get(k, 0)] for k in d_1_2] )#left outer join
'''