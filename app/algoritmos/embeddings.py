"""Modelos de IA (embeddings) para el Requerimiento 1.

No hay acceso a internet para descargar modelos preentrenados, asi que se entrenan
dos modelos de embeddings con el texto completo de los 20 articulos:
  1) Word2Vec Skip-gram con muestreo negativo (SGNS), escrito con numpy.
  2) PPMI + SVD (la idea matematica detras de GloVe / LSA), escrito con numpy.
El vector de un abstract = promedio de los vectores de sus palabras.
"""
import math, os
import numpy as np
from collections import Counter

DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

# ------------------------------------------------------------------ vocabulario
def construir_vocab(textos_tokens, min_count=5):
    c = Counter()
    for t in textos_tokens:
        c.update(t)
    palabras = [w for w, n in c.most_common() if n >= min_count]
    idx = {w: i for i, w in enumerate(palabras)}
    cuentas = np.array([c[w] for w in palabras], dtype=np.float64)
    return palabras, idx, cuentas

def _pares(secuencias_ids, ventana):
    """Pares (centro, contexto) dentro de la ventana."""
    C, X = [], []
    for s in secuencias_ids:
        s = np.asarray(s)
        for d in range(1, ventana + 1):
            if len(s) <= d:
                break
            C.append(s[:-d]); X.append(s[d:])      # centro a la izquierda
            C.append(s[d:]); X.append(s[:-d])      # centro a la derecha
    return np.concatenate(C), np.concatenate(X)

# ------------------------------------------------------------- 1) Word2Vec SGNS
def sigmoide(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))

def entrenar_sgns(textos_tokens, dim=50, ventana=5, negativos=5, epocas=5,
                  lr0=0.05, min_count=5, batch=1024, semilla=7):
    rng = np.random.default_rng(semilla)
    palabras, idx, cuentas = construir_vocab(textos_tokens, min_count)
    V = len(palabras)
    seqs = [[idx[w] for w in t if w in idx] for t in textos_tokens]
    C, X = _pares(seqs, ventana)
    prob = cuentas ** 0.75
    prob /= prob.sum()
    acum = np.cumsum(prob)
    W_in = (rng.random((V, dim)) - 0.5) / dim
    W_out = np.zeros((V, dim))
    n = len(C)
    total_pasos = epocas * math.ceil(n / batch)
    paso = 0
    perdidas = []
    for ep in range(epocas):
        orden = rng.permutation(n)
        perdida_ep, nb = 0.0, 0
        for ini in range(0, n, batch):
            sel = orden[ini:ini + batch]
            c, o = C[sel], X[sel]
            neg = np.searchsorted(acum, rng.random((len(sel), negativos)))
            neg = np.minimum(neg, V - 1)
            lr = lr0 * max(0.05, 1 - paso / total_pasos)
            v = W_in[c]; u_pos = W_out[o]; u_neg = W_out[neg]
            s_pos = sigmoide(np.sum(v * u_pos, axis=1))
            s_neg = sigmoide(np.einsum("bd,bkd->bk", v, u_neg))
            g_pos = (s_pos - 1.0)[:, None]
            dv = g_pos * u_pos + np.einsum("bk,bkd->bd", s_neg, u_neg)
            np.add.at(W_out, o, -lr * g_pos * v)
            np.add.at(W_out, neg, -lr * s_neg[:, :, None] * v[:, None, :])
            np.add.at(W_in, c, -lr * dv)
            perdida_ep += float(-np.mean(np.log(s_pos + 1e-9)) - np.mean(np.sum(np.log(1 - s_neg + 1e-9), axis=1)))
            nb += 1; paso += 1
        perdidas.append(perdida_ep / nb)
    return palabras, W_in, perdidas

# ------------------------------------------------------------------ 2) PPMI + SVD
def entrenar_ppmi_svd(textos_tokens, dim=50, ventana=5, min_count=8):
    palabras, idx, _ = construir_vocab(textos_tokens, min_count)
    V = len(palabras)
    M = np.zeros((V, V))
    for t in textos_tokens:
        s = [idx[w] for w in t if w in idx]
        for d in range(1, ventana + 1):
            a, b = np.array(s[:-d]), np.array(s[d:])
            if len(a) == 0:
                continue
            np.add.at(M, (a, b), 1.0)
            np.add.at(M, (b, a), 1.0)
    total = M.sum()
    fila = M.sum(axis=1, keepdims=True)
    col = (M.sum(axis=0, keepdims=True)) ** 0.75
    col = col / col.sum() * total
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log((M * total) / (fila * col + 1e-12) + 1e-12)
    ppmi = np.maximum(pmi, 0)
    ppmi[M == 0] = 0
    U, S, _ = np.linalg.svd(ppmi, full_matrices=False)
    W = U[:, :dim] * np.sqrt(S[:dim])
    return palabras, W

# ------------------------------------------------------------ vectores de documento
def vector_documento(tokens, palabras_idx, W, idf=None):
    """Promedio (opcionalmente ponderado por idf) de los vectores de las palabras conocidas."""
    acc = np.zeros(W.shape[1]); peso = 0.0
    for t in tokens:
        i = palabras_idx.get(t)
        if i is None:
            continue
        p = idf.get(t, 1.0) if idf else 1.0
        acc += p * W[i]; peso += p
    return acc / peso if peso > 0 else acc

def guardar(nombre, palabras, W):
    np.savez_compressed(os.path.join(DATA, nombre), palabras=np.array(palabras), W=W)

def cargar(nombre):
    z = np.load(os.path.join(DATA, nombre), allow_pickle=False)
    palabras = [str(p) for p in z["palabras"]]
    return palabras, {p: i for i, p in enumerate(palabras)}, z["W"]

def vecinos(palabra, palabras, idx, W, k=6):
    if palabra not in idx:
        return []
    v = W[idx[palabra]]
    N = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-12)
    sims = N @ (v / (np.linalg.norm(v) + 1e-12))
    orden = np.argsort(-sims)
    return [(palabras[i], float(sims[i])) for i in orden if palabras[i] != palabra][:k]
