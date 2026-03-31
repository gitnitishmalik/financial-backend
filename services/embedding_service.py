class EmbeddingService:
    def __init__(self):
        self._model = None  # don't load on startup

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._model

    def embed(self, texts):
        return self.model.encode(texts)