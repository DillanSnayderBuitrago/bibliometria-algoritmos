"""Clustering jerarquico aglomerativo desde cero (Requerimiento 2).

Se implementan tres criterios de enlace (linkage): Single, Complete y Average.
La salida es una lista de fusiones (a, b, distancia, tamano) al estilo 'nuevo_id = n + paso'.
"""
import math

def distancias_desde_similitud(S):
    n = len(S)
    return [[0.0 if i == j else 1.0 - S[i][j] for j in range(n)] for i in range(n)]

def distancia_clusters(D, A, B, metodo):
    vals = [D[i][j] for i in A for j in B]
    if metodo == "single":
        return min(vals)
    if metodo == "complete":
        return max(vals)
    if metodo == "average":
        return sum(vals) / len(vals)
    raise ValueError(metodo)

def aglomerativo(D, metodo):
    n = len(D)
    clusters = {i: [i] for i in range(n)}
    fusiones = []
    sig = n
    while len(clusters) > 1:
        mejor = None
        ids = sorted(clusters)
        for x in range(len(ids)):
            for y in range(x + 1, len(ids)):
                d = distancia_clusters(D, clusters[ids[x]], clusters[ids[y]], metodo)
                if mejor is None or d < mejor[0] - 1e-15:
                    mejor = (d, ids[x], ids[y])
        d, a, b = mejor
        clusters[sig] = clusters.pop(a) + clusters.pop(b)
        fusiones.append((a, b, d, len(clusters[sig])))
        sig += 1
    return fusiones

def cortar(fusiones, n, k):
    """Devuelve etiquetas (0..k-1) al cortar el arbol en k clusters."""
    miembros = {i: [i] for i in range(n)}
    for paso, (a, b, d, t) in enumerate(fusiones[: n - k]):
        miembros[n + paso] = miembros.pop(a) + miembros.pop(b)
    etiquetas = [0] * n
    for lab, (_, lista) in enumerate(sorted(miembros.items())):
        for i in lista:
            etiquetas[i] = lab
    return etiquetas

# ------------------------------------------------------- metricas internas
def silhouette(D, etiquetas):
    """Silhouette promedio: s = (b - a) / max(a, b). Cercano a 1 = clusters compactos y separados."""
    n = len(D)
    labs = sorted(set(etiquetas))
    if len(labs) < 2:
        return 0.0
    total = 0.0
    for i in range(n):
        propios = [j for j in range(n) if etiquetas[j] == etiquetas[i] and j != i]
        if not propios:
            continue  # cluster de un solo elemento: s = 0
        a = sum(D[i][j] for j in propios) / len(propios)
        b = min(sum(D[i][j] for j in range(n) if etiquetas[j] == l) / sum(1 for j in range(n) if etiquetas[j] == l)
                for l in labs if l != etiquetas[i])
        total += (b - a) / max(a, b) if max(a, b) > 0 else 0.0
    return total / n

def distancias_cofeneticas(fusiones, n):
    """Distancia cofenetica(i,j) = altura a la que i y j se unen en el arbol."""
    C = [[0.0] * n for _ in range(n)]
    miembros = {i: [i] for i in range(n)}
    for paso, (a, b, d, t) in enumerate(fusiones):
        for i in miembros[a]:
            for j in miembros[b]:
                C[i][j] = C[j][i] = d
        miembros[n + paso] = miembros.pop(a) + miembros.pop(b)
    return C

def pearson(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den = math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
    return num / den if den else 0.0

def cophenetic_corr(D, fusiones):
    """Correlacion entre las distancias originales y las del arbol (1 = el arbol conserva bien las distancias)."""
    n = len(D)
    C = distancias_cofeneticas(fusiones, n)
    x = [D[i][j] for i in range(n) for j in range(i + 1, n)]
    y = [C[i][j] for i in range(n) for j in range(i + 1, n)]
    return pearson(x, y)

# --------------------------------------------------------------- dendrograma
def orden_hojas(fusiones, n):
    hijos = {n + p: (a, b) for p, (a, b, d, t) in enumerate(fusiones)}
    def rec(nodo):
        if nodo < n:
            return [nodo]
        a, b = hijos[nodo]
        return rec(a) + rec(b)
    return rec(n + len(fusiones) - 1)

def dibujar_dendrograma(fusiones, etiquetas, titulo, ax):
    """Dibuja el dendrograma con lineas de matplotlib (calculo de posiciones propio)."""
    n = len(etiquetas)
    orden = orden_hojas(fusiones, n)
    x = {hoja: i for i, hoja in enumerate(orden)}
    y = {hoja: 0.0 for hoja in orden}
    for p, (a, b, d, t) in enumerate(fusiones):
        nodo = n + p
        ax.plot([x[a], x[a], x[b], x[b]], [y[a], d, d, y[b]], color="#1f4e79", lw=1.4)
        x[nodo] = (x[a] + x[b]) / 2
        y[nodo] = d
    ax.set_xticks(range(n))
    ax.set_xticklabels([etiquetas[h] for h in orden], rotation=90, fontsize=9)
    ax.set_ylabel("Distancia (1 - coseno)")
    ax.set_title(titulo)
    ax.grid(axis="y", alpha=0.3)
