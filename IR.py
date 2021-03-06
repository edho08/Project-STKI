import TextMining
import Ranking
import numpy as np
import concurrent
from sklearn.cluster import KMeans
import time
class IR:
    def __init__(self, folder):
        #Load file hasil preprocessing
        self.corpus = TextMining.load_from_file(folder+"/tfidf")
        self.ranker = Ranking.load_from_file(folder+"/pagerank")
        
        #simpan file raw txt dan pdf
        self.text_folder = folder+"/text"
        self.pdf_folder = folder+"/pdf_pages"
        
        #Normalisasi TFIDF
        tfidf = self.corpus.TFIDF
        with concurrent.futures.ThreadPoolExecutor() as executor:
            tfidf = np.array(list(executor.map(lambda x : x/np.linalg.norm(x), tfidf)))
        self.corpus.TFIDF = tfidf
        
    def query(self, query, N=20):
        #Dapatkan TFIDF dari query
        tf_idf_query = np.array(self.corpus.getQuery([[query]], stemmer = [TextMining.potter_stemmer]))
        
        #ambil TFIDF dan rank
        tfidf = self.corpus.TFIDF
        rank = np.array([self.ranker.rank])
        
        #Hitung cosine similarity antara query dengan semua dokumen
        cosine_vec = (tfidf @ tf_idf_query.T).T
        
        #periksa apabila tidak ada dokumen relevan maka proses tidak perlu dilanjutkan
        if all(v == 0 for v in cosine_vec[0]):
            return None
        
        #Sorting dokumen berdasarkan nilai similarity dan ambil dokumen sebanyak N
        #matrix retrieved berisi nomor halaman -1
        retrieved = np.argsort(cosine_vec)[0, -N:]

        # Untuk setiap dokumen relevan pada vector retrieved, ambil rank dan similarity        
        retrieved_cosine_vec = np.array([cosine_vec[0, retrieved[i]] for i in range(len(retrieved))])
        rank_vec = [rank[0, retrieved[i]] for i in range(len(retrieved))]
        
        # Kalikan rank dan similarity untuk proses ranking
        score_vec = retrieved_cosine_vec*rank_vec
        
        # sort hasil ranking
        ranked_retrieval = np.argsort(score_vec)
        
        # ambil score ranking untuk masing-masing dokumen
        ranked_score_vec = [score_vec[ranked_retrieval[i]] for i in range(len(ranked_retrieval))]
        
        # perbaiki nomor halaman dokumen dan buang halaman dengan nilai score 0 
        returned_doc = [retrieved[ranked_retrieval[i]]+1 for i in range(len(ranked_retrieval))]
        returned_doc = list(filter(lambda X : X[1]>0, list(zip(returned_doc, ranked_score_vec))))
        
        #Kembalikan dokumen yang sudah diambil ke USER
        return [returned_doc[len(returned_doc)-i-1] for i in range(len(returned_doc))]
    
def IRTest(texts, ir, num_cluster, percentage):
    #Buat cluster menggunakan algoritma kmeans
    kmeans = KMeans(n_clusters=num_cluster).fit(ir.corpus.TFIDF)
    
    #gabung antara cluster dengan halaman
    label = list(zip(kmeans.labels_, range(len(texts))))
    
    #vector test untuk menyimpan nilai precision, recall dan fmeasure
    test = []
    
    #untuk setiap cluster
    for cluster in range(num_cluster):
        #ambil dokumen yang berada pada cluster
        label_doc = list(filter(lambda x : x[0] == cluster, label))
        
        #ambil raw text dari dokumen dalam cluster
        label_text = [texts[label_doc[i][1]] for i in range(len(label_doc))]
        
        #ambil tfidf dokumen pada cluster
        label_tfidf = np.array([ir.corpus.TFIDF[label_doc[i][1]] for i in range(len(label_doc))])
        
        #lakukan peringkasan dengan algoritma textrank
        summa = Ranking.TextRank(label_tfidf).summarize(label_text, percentage)
        
        #retrive doc dengan query summa
        start = time.time()
        retrieved = [doc[0] for doc in ir.query(summa, len(label_doc))]
        print("time=%f"%(time.time() - start))
        
        #hitung confusion matrix
        dic_label = {page[1]:True for page in label_doc}
        dic_retrieved = {page:True for page in retrieved}
        TP = sum([1 if page in dic_label else 0 for page in retrieved])
        FP = sum([1 if page not in dic_label else 0 for page in retrieved])
        FN = sum([1 if page not in dic_retrieved else 0 for page in label_doc])

        #hitung recall dan precision
        precision = float(TP) / (TP + FP)
        recall = float(TP) / (TP + FN)
        fmeasure = 2 * (precision * recall / (precision + recall))
        test.append([precision, recall, fmeasure])
    return test
        
'''        
data=[]
for i in range(1,800):
    data.append([open(str.format("deeplearning/text/page-%d.txt" % i), "r").read()])
ir = IR("deeplearning")

#test
cluster = [4, 6, 8, 10]
percentage = [0.1, 0.3, 0.5]

print("test start")
start = time.time()

for c in cluster:
    for p in percentage:
        print("cluster = %f, percentage=%f" % (c, p))
        summary = IRTest(data, ir, c, p)
        summary = np.average(summary, axis=0)
        print(summary)
        
print("test end")
print(time.time()- start) 
'''       
'''        
ir = IR("deeplearning")
#print(ir.query([["goodfellow"]]))   #penulis buku
print(ir.query([["neuroscience"]]))   #acknowledgement
#print(ir.query([["Deep Neural Network"]]))   
#print(ir.query([["Knowledge Representation"]]))
#print(ir.query([["Neuroscience"]]))
'''