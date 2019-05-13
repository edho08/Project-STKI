import TextMining
import numpy as np
corpus = TextMining.load_from_file("deeplearning/tfidf")
tfidf = np.array(corpus.TFIDF)

print("SVD")
U, S, Vt = np.linalg.svd(tfidf)
print(S)