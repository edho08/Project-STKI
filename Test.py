# -*- coding: utf-8 -*-
import IR
import numpy
import matplotlib.pyplot as plt
ir = IR.IR("deeplearning")
corpus = ir.corpus
"""
    Dibawah ini merupakan tes coverage dari IR.
    query yang digunakan adalah token random dari korpus
"""
whole_hit = numpy.zeros((len(corpus.TFIDF)))
#ulang hingga 10x
for i in range(20):
    query = ""
    coverage = [0] * len(corpus.TFIDF)
    hit = [0] * len(corpus.TFIDF)
    for c in range(50):
        for j in range(i+1):
            query = corpus.unique_terms.collect()[int(numpy.random.uniform(0, len(corpus.unique_terms.collect())))][0]+ " " + query  
        result = ir.query([[query]])
        for res in result:
            coverage[res[0]-1] = 1
            hit[res[0]-1] += 1
    whole_hit += hit
    print("token = %d" %i)
    print("coverage = %f" % (sum(coverage)/len(coverage)))
    print("hit = %f" % (numpy.sum(hit)))
    
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = numpy.exp(x - numpy.max(x))
    return e_x / e_x.sum(axis=0) # only difference

whole_hit_sum = whole_hit / sum(whole_hit)
labels = ["page " + str(i+1) for i in range(len(corpus.TFIDF))]
print(whole_hit_sum)