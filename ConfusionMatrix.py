import numpy as np

def confusion_matrix(prediction, groundTruth):
    klas = {}
    i = 0
    for data in groundTruth:
        if data not in klas:
           klas[data] = i
           i+=1
    confusion_matrix = np.zeros((len(klas), len(klas)))
    for i in range(len(prediction)):
        x = klas[prediction[i]]
        y = klas[groundTruth[i]]
        confusion_matrix[x,y] += 1
    return klas, confusion_matrix

def TP(cf):
    tp = [0] * len(cf)
    for i in range(len(tp)):
        tp[i] = cf[i,i]
    return np.array(tp)

def FP(cf):
    return [sum(cf[i]) for i in range(len(cf))] - TP(cf)

def FN(cf):
    return [sum(cf[:,i]) for i in range(len(cf))] - TP(cf)

def TN(cf):
    return [sum(sum(cf))] * len(cf) - TP(cf) - FP(cf) - FN(cf)

def cf_summary(cf):
    return TP(cf), FP(cf), FN(cf), TN(cf)

def summary(cf):
    tp, fp, fn, tn = cf_summary(cf)
    accuracy = sum(tp / (tp+fp+fn+tn))
    precision = tp / (tp+fp)
    recall = tp / (tp+fn)
    return accuracy, precision, recall
