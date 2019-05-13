from Distance import Max
import numpy as np
import concurrent
import os
import math
class CosinePageRank:
    def __init__(self, tfidf, iteration=50):
        if tfidf is None:
            return 
        with concurrent.futures.ThreadPoolExecutor() as executor:
            tfidf = np.array(list(executor.map(lambda x : x/np.linalg.norm(x), tfidf)))
        cosine_H = tfidf @ tfidf.T
        self.H = np.array([[1-abs(i-j)/len(cosine_H) for j in range(len(cosine_H[i]))] for i in range(len(cosine_H))]) * cosine_H
        for i in range(len(self.H)):
            self.H[i,i] = 0
        for i in range(len(self.H)):
            for j in range(len(self.H[i])):
                self.H[i,j] = self.H[i,j] if self.H[i,j] > 0.001 else 0
        self.H = Max(self.H, 1).T       
        self.rank = self.calculateRank(iteration)
        
        
    def calculateRank(self, iteration):
        rank = np.array([1/len(self.H)] * len(self.H))
        H = self.H.copy()
        for i in range(iteration):
            H = H @ H
        return H @ rank
    
    def toFile(self, folder):
        try:
            os.makedirs(folder)
        except :
            pass
        print("saving H")
        with open(folder+'/H', 'w') as file:
            for h in self.H:
                    for h1 in h :   
                        file.write(str(h1)+',')
                    file.write('\n')
        print("saving rank")
        with open(folder+'/rank', 'w') as file:
            for r in self.rank:
                file.write(str(r)+"\n")
            file.write('\n')
    
class TextRank(CosinePageRank):
    def summarize(self, text, percentage=0.3):
        count = math.ceil(percentage * len(text))
        ranked_doc = np.argsort(-self.rank)[:count]
        summarized = ' '.join([text[r][0] for r in ranked_doc])
        return summarized

        
def load_from_file(folder):
    ranker = CosinePageRank(None)
    print("loaing rank")
    ra = open(str.format("%s/rank"%folder), 'r').readlines()
    un = []
    for u in ra:
        try:
            un.append(float(u.strip()))
        except:
            pass
    ranker.rank = un
    
    print("loading H")
    h = open(str.format("%s/H"%folder), 'r').readlines()
    un = []
    for u in h:
        a = ((u.strip().split(',')))[:-1]
        a = [float(b) for b in a]
        un.append(a)
    ranker.H = un
    return ranker