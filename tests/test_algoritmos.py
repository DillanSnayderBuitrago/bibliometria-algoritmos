import sys, random, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
from algoritmos import busqueda as B, similitud as S, clustering as C

def ocurrencias_ref(t, p):
    # referencia independiente con str.find SOLO para validar (no se usa en la app)
    res, i = [], t.find(p)
    while i != -1:
        res.append(i); i = t.find(p, i + 1)
    return res

random.seed(1)
n_ok = 0
for _ in range(3000):
    alf = random.choice(["ab", "abc", "abcd", "acgt"])
    t = "".join(random.choice(alf) for _ in range(random.randint(1, 60)))
    p = "".join(random.choice(alf) for _ in range(random.randint(1, 6)))
    ref = ocurrencias_ref(t, p)
    for nombre, f in B.ALGORITMOS.items():
        assert f(t, p)["posiciones"] == ref, (nombre, t, p)
    assert B.fuerza_bruta(t, p)["posiciones"] == ref
    n_ok += 1
# Rabin-Karp con primo pequeno (muchas colisiones) tambien debe ser correcto
for _ in range(500):
    t = "".join(random.choice("abc") for _ in range(50)); p = "".join(random.choice("abc") for _ in range(3))
    assert B.rabin_karp(t, p, 256, 101)["posiciones"] == ocurrencias_ref(t, p)
print("busqueda OK:", n_ok, "casos aleatorios x 4 algoritmos")

# Levenshtein clasico: kitten/sitting = 3
assert S.levenshtein(list("kitten"), list("sitting")) == 3
assert S.levenshtein([], list("abc")) == 3
assert abs(S.jaccard(["a","b"],["b","c"]) - 1/3) < 1e-12
print("similitud OK")

# Clustering contra scipy (solo validacion)
import numpy as np
from scipy.cluster.hierarchy import linkage, cophenet
from scipy.spatial.distance import squareform
rng = np.random.default_rng(3)
X = rng.random((20, 5))
D = np.linalg.norm(X[:, None] - X[None], axis=2)
for m in ["single", "complete", "average"]:
    mine = C.aglomerativo(D.tolist(), m)
    Z = linkage(squareform(D, checks=False), method=m)
    assert np.allclose(sorted(f[2] for f in mine), sorted(Z[:, 2])), m
    assert abs(C.cophenetic_corr(D.tolist(), mine) - cophenet(Z, squareform(D, checks=False))[0]) < 1e-9, m
print("clustering OK (alturas y CPCC coinciden con scipy)")
