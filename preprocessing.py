import TextMining
import nltk
import numpy as np
import Ranking
data = []
for i in range(1,800):
    data.append([open(str.format("deeplearning/text/page-%d.txt" % i), "r").read()])

stemmer = nltk.stem.PorterStemmer()
data = TextMining.Dokumen_Cluster(data, stemmer = [np.vectorize(stemmer.stem)], stopword=[TextMining.english_stopwords], verbose=True)
data.toFile("deeplearning/tfidf")
rank = Ranking.CosinePageRank(data.TFIDF)
rank.toFile("deeplearning/pagerank")