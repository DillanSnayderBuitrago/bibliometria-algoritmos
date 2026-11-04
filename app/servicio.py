"""Capa de servicio: junta corpus + algoritmos. La usan la app web y los experimentos."""
import numpy as np
from algoritmos.corpus import cargar_corpus
from algoritmos.preprocesamiento import preprocesar
from algoritmos import similitud as S, embeddings as E, clustering as C

NOMBRES_SIM = ["Levenshtein", "Needleman-Wunsch", "Jaccard", "TF-IDF + Coseno",
               "Word2Vec (SGNS) + Coseno", "PPMI-SVD + Euclidiana"]
TIPO = {n: ("Clasico" if i < 4 else "IA (embeddings)") for i, n in enumerate(NOMBRES_SIM)}

class Servicio:
    def __init__(self):
        self.docs = cargar_corpus()
        self.tokens = [preprocesar(d["abstract"]) for d in self.docs]
        self.idf = S.calcular_idf(self.tokens)
        self.pal1, self.idx1, self.W1 = E.cargar("w2v_sgns.npz")
        self.pal2, self.idx2, self.W2 = E.cargar("ppmi_svd.npz")
        # vector del abstract = promedio ponderado (idf) de sus palabras
        self.v1_raw = [E.vector_documento(t, self.idx1, self.W1, self.idf) for t in self.tokens]
        self.v2_raw = [E.vector_documento(t, self.idx2, self.W2, self.idf) for t in self.tokens]
        # centrado: se resta el vector promedio del corpus (quita la direccion comun a todos los abstracts)
        self.m1 = np.mean(self.v1_raw, axis=0); self.m2 = np.mean(self.v2_raw, axis=0)
        self.v1 = [v - self.m1 for v in self.v1_raw]
        self.v2 = [v - self.m2 for v in self.v2_raw]
        # normalizados (largo 1) para la distancia euclidiana
        self.v2n = [v / (np.linalg.norm(v) + 1e-12) for v in self.v2]
        self._cache = {}

    def sim(self, nombre, i, j):
        """Similitud en [0,1] entre los abstracts i y j (indices 0..19)."""
        a, b = self.tokens[i], self.tokens[j]
        if nombre == "Levenshtein":
            return S.similitud_levenshtein(a, b)
        if nombre == "Needleman-Wunsch":
            return S.similitud_nw(a, b)
        if nombre == "Jaccard":
            return S.jaccard(a, b)
        if nombre == "TF-IDF + Coseno":
            return S.similitud_tfidf_coseno(a, b, self.idf)
        if nombre == "Word2Vec (SGNS) + Coseno":
            return max(0.0, S.coseno_denso(self.v1[i], self.v1[j]))
        if nombre == "PPMI-SVD + Euclidiana":
            return 1 - S.euclidiana(self.v2n[i], self.v2n[j]) / 2   # d en [0,2] -> sim en [0,1]
        raise ValueError(nombre)

    def sim_cache(self, nombre, i, j):
        k = (nombre, min(i, j), max(i, j))
        if k not in self._cache:
            self._cache[k] = self.sim(nombre, i, j)
        return self._cache[k]

    def matriz_tfidf_coseno(self):
        n = len(self.docs)
        V = [S.vector_tfidf(t, self.idf) for t in self.tokens]
        return [[1.0 if i == j else S.coseno_sparse(V[i], V[j]) for j in range(n)] for i in range(n)]
