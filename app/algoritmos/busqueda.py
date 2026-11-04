"""Busqueda exacta de patrones (Requerimiento 3). Todo desde cero: no se usa find(), index(), in ni re.

Cada algoritmo devuelve un diccionario con:
  posiciones   : lista de indices (base 0) donde empieza cada coincidencia
  comparaciones: numero de comparaciones de caracteres realizadas
  (extras propios de cada algoritmo)
"""

def minusculas(s):
    return s.lower()

# --------------------------------------------------------------- fuerza bruta (referencia)
def fuerza_bruta(texto, patron):
    n, m = len(texto), len(patron)
    pos, comp = [], 0
    for i in range(n - m + 1):
        j = 0
        while j < m:
            comp += 1
            if texto[i + j] != patron[j]:
                break
            j += 1
        if j == m:
            pos.append(i)
    return {"posiciones": pos, "comparaciones": comp}

# ---------------------------------------------------------------------- KMP
def tabla_lps(patron):
    """lps[i] = longitud del prefijo propio mas largo de patron[:i+1] que tambien es sufijo."""
    m = len(patron)
    lps = [0] * m
    k = 0
    for i in range(1, m):
        while k > 0 and patron[i] != patron[k]:
            k = lps[k - 1]
        if patron[i] == patron[k]:
            k += 1
        lps[i] = k
    return lps

def kmp(texto, patron):
    n, m = len(texto), len(patron)
    if m == 0 or m > n:
        return {"posiciones": [], "comparaciones": 0, "lps": []}
    lps = tabla_lps(patron)
    pos, comp = [], 0
    j = 0
    for i in range(n):
        while True:
            comp += 1
            if texto[i] == patron[j]:
                j += 1
                break
            if j == 0:
                break
            j = lps[j - 1]
        if j == m:
            pos.append(i - m + 1)
            j = lps[j - 1]
    return {"posiciones": pos, "comparaciones": comp, "lps": lps}

# --------------------------------------------------------------- Boyer-Moore
def tabla_caracter_malo(patron):
    """Ultima posicion de cada caracter dentro del patron."""
    return {c: i for i, c in enumerate(patron)}

def tabla_buen_sufijo(patron):
    """Regla del buen sufijo (version fuerte). shift[j] = desplazamiento si falla en j-1."""
    m = len(patron)
    shift = [0] * (m + 1)
    borde = [0] * (m + 1)
    i, j = m, m + 1
    borde[i] = j
    while i > 0:
        while j <= m and patron[i - 1] != patron[j - 1]:
            if shift[j] == 0:
                shift[j] = j - i
            j = borde[j]
        i -= 1; j -= 1
        borde[i] = j
    j = borde[0]
    for i in range(m + 1):
        if shift[i] == 0:
            shift[i] = j
        if i == j:
            j = borde[j]
    return shift

def boyer_moore(texto, patron):
    n, m = len(texto), len(patron)
    if m == 0 or m > n:
        return {"posiciones": [], "comparaciones": 0, "malo": {}, "sufijo": []}
    malo = tabla_caracter_malo(patron)
    sufijo = tabla_buen_sufijo(patron)
    pos, comp = [], 0
    s = 0
    while s <= n - m:
        j = m - 1
        while j >= 0:
            comp += 1
            if patron[j] != texto[s + j]:
                break
            j -= 1
        if j < 0:
            pos.append(s)
            s += sufijo[0]
        else:
            s += max(sufijo[j + 1], j - malo.get(texto[s + j], -1))
    return {"posiciones": pos, "comparaciones": comp, "malo": malo, "sufijo": sufijo}

# ---------------------------------------------------------------- Rabin-Karp
def rabin_karp(texto, patron, base=256, primo=1000000007):
    """Hash rodante: h(s[i+1..i+m]) = (base*(h(s[i..i+m-1]) - s[i]*base^(m-1)) + s[i+m]) mod primo."""
    n, m = len(texto), len(patron)
    if m == 0 or m > n:
        return {"posiciones": [], "comparaciones": 0, "colisiones": 0, "hash_patron": 0}
    pot = 1
    for _ in range(m - 1):
        pot = (pot * base) % primo
    hp = ht = 0
    for i in range(m):
        hp = (base * hp + ord(patron[i])) % primo
        ht = (base * ht + ord(texto[i])) % primo
    pos, comp, colisiones = [], 0, 0
    for i in range(n - m + 1):
        if hp == ht:
            j = 0
            while j < m:
                comp += 1
                if texto[i + j] != patron[j]:
                    break
                j += 1
            if j == m:
                pos.append(i)
            else:
                colisiones += 1
        if i < n - m:
            ht = (base * (ht - ord(texto[i]) * pot) + ord(texto[i + m])) % primo
    return {"posiciones": pos, "comparaciones": comp, "colisiones": colisiones, "hash_patron": hp}

ALGORITMOS = {"KMP": kmp, "Boyer-Moore": boyer_moore, "Rabin-Karp": rabin_karp}
