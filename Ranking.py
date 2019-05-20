from Distance import Max
import numpy as np
import concurrent
import os
import math
class CosinePageRank:
    #construtor untuk class pageRank
    def __init__(self, tfidf, iteration=50):
        #Periksa apabila tfidf tidak disediakan user
        if tfidf is None:
            return 
        
        #Normalisasi TFIDF
        with concurrent.futures.ThreadPoolExecutor() as executor:
            tfidf = np.array(list(executor.map(lambda x : x/np.linalg.norm(x), tfidf)))
        
        #Hitung Cosine Similarity untuk sertiap dokumen terhadap setiap dokumen
        cosine_H = tfidf @ tfidf.T
        
        #Hitung kedekatan halaman antar dokumen. rumus 1 - absolut(halaman i - halaman j) / total_halaman
        self.H = np.array([[1-abs(i-j)/len(cosine_H) for j in range(len(cosine_H[i]))] for i in range(len(cosine_H))]) * cosine_H
        
        #Tidak ada self connection pada graf
        for i in range(len(self.H)):
            self.H[i,i] = 0
        
        #Putuskan koneksi antar node apabila nilai koneksi kurang dari threshold (0.001)
        for i in range(len(self.H)):
            for j in range(len(self.H[i])):
                self.H[i,j] = self.H[i,j] if self.H[i,j] > 0.001 else 0
                
        #simpan matrix H
        self.H = Max(self.H, 1).T 
        
        #Hitung dan simpan vector Rank
        self.rank = self.calculateRank(iteration)
        
        
    def calculateRank(self, iteration):
        #Nilai inisial rank untuk tiap dokumen. rumus 1/total_dokumen
        rank = np.array([1/len(self.H)] * len(self.H))
        
        #buat kopi matrix H
        H = self.H.copy()
        
        #lakukan power iteration sebanyak iteration.
        #Power iteration dibawah menghasilkan banyak perulangan 2 ^ iteration
        for i in range(iteration):
            H = H @ H
        
        #kembalikan nilai rank dengan mengkalikan matrix H iterasi ke iteration dengan vector rank
        return H @ rank
    
class TextRank(CosinePageRank):
    #class algoritma peringkasan text rank
    def summarize(self, text, percentage=0.3):
        # ambil banyak instan yang akan diekstrak.
        count = math.ceil(percentage * len(text))
        
        #sort rank dokumen secara descending. ambil dokumen dengan nilai rank tertinggi sebanyak count
        ranked_doc = np.argsort(-self.rank)[:count]
        
        #join string dari dokumen-dokumen yang telah diambil
        summarized = ' '.join([text[r][0] for r in ranked_doc])
        
        #kembalikan string hasil peringkasan
        return summarized


    def toFile(self, folder):
        #Menyimpan class ke file pada folder
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
    
       
def load_from_file(folder):
    #meload file dari folder
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