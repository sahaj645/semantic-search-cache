from .model_loader import get_model

def embed(texts):
    model=get_model()
    return model.encode(texts,normalize_embeddings=True)