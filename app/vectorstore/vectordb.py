import faiss
import numpy as np

class VectorDB:

    def __init__(self,dim):
        self.index=faiss.IndexFlatIP(dim)
        self.vectors=None
        self.docs=[]

    def add(self,embeddings,docs):
        if self.vectors is None:
            self.vectors=embeddings
        else:
            self.vectors=np.vstack((self.vectors,embeddings))

        self.index.add(embeddings)
        self.docs.extend(docs)

    def search(self,vector,k=5):
        D,I=self.index.search(vector,k)
        return [(self.docs[i],float(D[0][j])) for j,i in enumerate(I[0])]