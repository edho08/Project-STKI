import TextMining
import nltk
import numpy as np
import Ranking

#matrix data untuk menyimpan dokumen
data = []

#baca semua halaman dari folder deeplearning/text
for i in range(1,800):
    data.append([open(str.format("deeplearning/text/page-%d.txt" % i), "r").read()])

#buat potter stemmer
stemmer = nltk.stem.PorterStemmer()

#Hitung TFIDF dari data
data = TextMining.Dokumen_Cluster(data, stemmer = [np.vectorize(stemmer.stem)], stopword=[TextMining.english_stopwords], verbose=True)

#simpan TFIDF ke file
data.toFile("deeplearning/tfidf")

#Hitung rank
rank = Ranking.CosinePageRank(data.TFIDF)

#simpan rank ke file
rank.toFile("deeplearning/pagerank")