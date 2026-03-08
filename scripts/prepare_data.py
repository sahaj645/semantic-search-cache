from sklearn.datasets import fetch_20newsgroups
import pickle
import re

def clean_text(text):
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.lower().strip()

data = fetch_20newsgroups(remove=('headers','footers','quotes'))

docs = [clean_text(d) for d in data.data if len(d) > 50]

with open("data/newsgroups.pkl","wb") as f:
    pickle.dump(docs,f)

print("Saved",len(docs),"documents")