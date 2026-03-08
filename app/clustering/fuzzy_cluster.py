from sklearn.mixture import GaussianMixture
import numpy as np

class FuzzyCluster:

    def __init__(self,n_clusters=12):
        self.model=GaussianMixture(n_components=n_clusters,covariance_type='tied')

    def fit(self,embeddings):
        self.model.fit(embeddings)

    def get_distribution(self,embedding):
        probs=self.model.predict_proba(embedding)
        return probs[0]

    def dominant_cluster(self,embedding):
        probs=self.get_distribution(embedding)
        return int(np.argmax(probs))