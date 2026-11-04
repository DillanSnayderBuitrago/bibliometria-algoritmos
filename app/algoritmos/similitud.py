"""Algoritmos de similitud textual (Requerimiento 1) - version clasica.

Todos los algoritmos trabajan sobre listas de tokens (palabras ya preprocesadas).
"""
import math
from collections import Counter

# ---------------------------------------------------------------- Levenshtein
def matriz_levenshtein(a, b):
    """Llena la matriz de programacion dinamica de la distancia de edicion.
    D[i][j] = distancia entre los primeros i elementos de a y los primeros j de b."""
    n, m = len(a), len(b)
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        D[i][0] = i
    for j in range(m + 1):
        D[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            costo = 0 if a[i - 1] == b[j - 1] else 1
            D[i][j] = min(D[i - 1][j] + 1,        # borrar
                          D[i][j - 1] + 1,        # insertar
                          D[i - 1][j - 1] + costo)  # sustituir / igual
    return D

def levenshtein(a, b):
    return matriz_levenshtein(a, b)[len(a)][len(b)]

def similitud_levenshtein(a, b):
    """1 - distancia / longitud maxima  (1 = identicos, 0 = totalmente distintos)."""
    mx = max(len(a), len(b))
    return 1.0 if mx == 0 else 1 - levenshtein(a, b) / mx

# ------------------------------------------------------- Needleman-Wunsch
def matriz_needleman_wunsch(a, b, match=1, mismatch=-1, gap=-1):
    """Alineamiento global. F[i][j] = mejor puntaje de alinear a[:i] con b[:j]."""
    n, m = len(a), len(b)
    F = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        F[i][0] = i * gap
    for j in range(1, m + 1):
        F[0][j] = j * gap
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diag = F[i - 1][j - 1] + (match if a[i - 1] == b[j - 1] else mismatch)
            F[i][j] = max(diag, F[i - 1][j] + gap, F[i][j - 1] + gap)
    return F

def alinear_nw(a, b, match=1, mismatch=-1, gap=-1):
    """Devuelve (puntaje, alineamiento) usando traceback sobre la matriz."""
    F = matriz_needleman_wunsch(a, b, match, mismatch, gap)
    i, j = len(a), len(b)
    A, B = [], []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and F[i][j] == F[i - 1][j - 1] + (match if a[i - 1] == b[j - 1] else mismatch):
            A.append(a[i - 1]); B.append(b[j - 1]); i -= 1; j -= 1
        elif i > 0 and F[i][j] == F[i - 1][j] + gap:
            A.append(a[i - 1]); B.append("-"); i -= 1
        else:
            A.append("-"); B.append(b[j - 1]); j -= 1
    return F[len(a)][len(b)], A[::-1], B[::-1]

def limites_nw(n, m, match=1, mismatch=-1, gap=-1):
    """Mejor y peor puntaje posibles para secuencias de largo n y m.
    Mejor: todo lo comun coincide.  Peor: todo lo alineado es diferente (mas los huecos obligatorios)."""
    k, dif = min(n, m), abs(n - m)
    return match * k + gap * dif, mismatch * k + gap * dif

def similitud_nw(a, b, match=1, mismatch=-1, gap=-1):
    """Puntaje NW reescalado a [0,1]:  (S - S_peor) / (S_mejor - S_peor)."""
    if not a and not b:
        return 1.0
    F = matriz_needleman_wunsch(a, b, match, mismatch, gap)
    mejor, peor = limites_nw(len(a), len(b), match, mismatch, gap)
    return (F[len(a)][len(b)] - peor) / (mejor - peor)

# ------------------------------------------------------------------ Jaccard
def jaccard(a, b):
    A, B = set(a), set(b)
    union = len(A | B)
    return 1.0 if union == 0 else len(A & B) / union

# ---------------------------------------------------------- TF-IDF y coseno
def calcular_idf(lista_docs):
    """idf(t) = ln(N / df(t)) + 1 ; N = numero de documentos del corpus."""
    N = len(lista_docs)
    df = Counter()
    for d in lista_docs:
        df.update(set(d))
    return {t: math.log(N / c) + 1 for t, c in df.items()}

def vector_tfidf(tokens, idf):
    """tf = frecuencia / total de palabras; peso = tf * idf."""
    c = Counter(tokens)
    total = len(tokens)
    return {t: (f / total) * idf.get(t, 1.0) for t, f in c.items()}

def producto_punto(u, v):
    if len(u) > len(v):
        u, v = v, u
    return sum(x * v.get(t, 0.0) for t, x in u.items())

def norma(u):
    return math.sqrt(sum(x * x for x in u.values()))

def coseno_sparse(u, v):
    nu, nv = norma(u), norma(v)
    return 0.0 if nu == 0 or nv == 0 else producto_punto(u, v) / (nu * nv)

def similitud_tfidf_coseno(a, b, idf):
    return coseno_sparse(vector_tfidf(a, idf), vector_tfidf(b, idf))

# ------------------------------------------------------ vectores densos (IA)
def coseno_denso(u, v):
    num = sum(x * y for x, y in zip(u, v))
    nu = math.sqrt(sum(x * x for x in u)); nv = math.sqrt(sum(y * y for y in v))
    return 0.0 if nu == 0 or nv == 0 else num / (nu * nv)

def euclidiana(u, v):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(u, v)))
