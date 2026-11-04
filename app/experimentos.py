import sys, json, time, math, glob, os
sys.path.insert(0, "app")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from servicio import Servicio, NOMBRES_SIM
from algoritmos import similitud as S, clustering as C, busqueda as B, embeddings as E
from algoritmos.preprocesamiento import preprocesar

plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10})
sv = Servicio(); n = 20
R = {}

# ============================================================= REQUERIMIENTO 1
pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
ia, ib = 11, 19                          # abstracts 12 y 20
a, b = sv.tokens[ia], sv.tokens[ib]
r1 = {"ia": ia + 1, "ib": ib + 1, "len_a": len(a), "len_b": len(b),
      "titulo_a": sv.docs[ia]["title"], "titulo_b": sv.docs[ib]["title"],
      "abs_a": sv.docs[ia]["abstract"], "abs_b": sv.docs[ib]["abstract"], "tok_a": a, "tok_b": b}
fa, fb = a[:8], b[:8]
r1["frag_a"], r1["frag_b"] = fa, fb
r1["lev_mat"] = S.matriz_levenshtein(fa, fb)
r1["lev_frag"] = r1["lev_mat"][8][8]
r1["lev_full"] = S.levenshtein(a, b)
r1["lev_sim"] = S.similitud_levenshtein(a, b)
r1["nw_mat"] = S.matriz_needleman_wunsch(fa, fb)
sc, A, Bb = S.alinear_nw(fa, fb)
r1["nw_frag_score"], r1["nw_align_a"], r1["nw_align_b"] = sc, A, Bb
Ff = S.matriz_needleman_wunsch(a, b)
r1["nw_full"] = Ff[len(a)][len(b)]
r1["nw_limites"] = S.limites_nw(len(a), len(b))
r1["nw_sim"] = S.similitud_nw(a, b)
SA, SB = set(a), set(b)
r1["jac"] = {"A": len(SA), "B": len(SB), "inter": sorted(SA & SB), "n_inter": len(SA & SB), "union": len(SA | SB), "valor": S.jaccard(a, b)}
va, vb = S.vector_tfidf(a, sv.idf), S.vector_tfidf(b, sv.idf)
from collections import Counter
ca, cb = Counter(a), Counter(b)
filas = []
for t in sorted(SA & SB):
    filas.append({"t": t, "fa": ca[t], "fb": cb[t], "tfa": ca[t] / len(a), "tfb": cb[t] / len(b), "idf": sv.idf[t], "wa": va[t], "wb": vb[t], "prod": va[t] * vb[t]})
r1["tfidf_filas"] = filas
r1["tfidf_dot"] = S.producto_punto(va, vb)
r1["tfidf_na"], r1["tfidf_nb"] = S.norma(va), S.norma(vb)
r1["tfidf_cos"] = S.coseno_sparse(va, vb)
r1["tfidf_ang"] = math.degrees(math.acos(r1["tfidf_cos"]))
# embeddings
d = 5
for key, raw, cen, nom in (("w2v", sv.v1_raw, sv.v1, "Word2Vec"), ("ppmi", sv.v2_raw, sv.v2, "PPMI")):
    u, v = cen[ia], cen[ib]
    r1[key] = {"u5": [float(x) for x in u[:d]], "v5": [float(x) for x in v[:d]],
               "raw_u5": [float(x) for x in raw[ia][:d]], "raw_v5": [float(x) for x in raw[ib][:d]],
               "cos": S.coseno_denso(u, v), "cos_raw": S.coseno_denso(raw[ia], raw[ib]),
               "eucl_raw": S.euclidiana(raw[ia], raw[ib]),
               "dot": float(np.dot(u, v)), "nu": float(np.linalg.norm(u)), "nv": float(np.linalg.norm(v))}
un, vn = sv.v2n[ia], sv.v2n[ib]
r1["ppmi"]["eucl_norm"] = S.euclidiana(un, vn)
r1["ppmi"]["sim"] = 1 - r1["ppmi"]["eucl_norm"] / 2
r1["w2v"]["sim"] = max(0.0, r1["w2v"]["cos"])
r1["dim"] = int(sv.W1.shape[1])
# vecinos de palabras clave (para explicar los modelos)
r1["vecinos"] = {w: {"w2v": E.vecinos(w, sv.pal1, sv.idx1, sv.W1, 5), "ppmi": E.vecinos(w, sv.pal2, sv.idx2, sv.W2, 5)}
                 for w in ["assessment", "review", "generative", "teacher"]}
# tabla comparativa de pares
def fila_par(i, j):
    return {"par": (i + 1, j + 1), **{nm: float(sv.sim_cache(nm, i, j)) for nm in NOMBRES_SIM}}
r1["pares"] = [fila_par(ia, ib), fila_par(14, 15), fila_par(2, 8), fila_par(10, 15)]
# rankings: pares mas similares segun cada modelo
r1["top5"] = {nm: [(p[0] + 1, p[1] + 1, float(sv.sim_cache(nm, *p))) for p in sorted(pairs, key=lambda p: -sv.sim_cache(nm, *p))[:5]] for nm in NOMBRES_SIM}
# correlacion entre metodos (Pearson sobre las 190 parejas)
M = np.array([[sv.sim_cache(nm, *p) for p in pairs] for nm in NOMBRES_SIM])
r1["corr"] = np.corrcoef(M).tolist()
r1["medias"] = {nm: float(M[k].mean()) for k, nm in enumerate(NOMBRES_SIM)}
# tiempos de cada algoritmo para el caso
tiempos = {}
for nm in NOMBRES_SIM:
    t0 = time.perf_counter()
    for _ in range(3):
        sv.sim(nm, ia, ib)
    tiempos[nm] = (time.perf_counter() - t0) / 3 * 1000
r1["tiempos_ms"] = tiempos
R["r1"] = r1
# figura: barras del caso
fig, ax = plt.subplots(figsize=(7.2, 3.4))
vals = [r1["pares"][0][nm] for nm in NOMBRES_SIM]
cols = ["#7f7f7f"] * 4 + ["#1f4e79"] * 2
labs = ["Levenshtein", "Needleman-\nWunsch", "Jaccard", "TF-IDF +\nCoseno", "Word2Vec +\nCoseno", "PPMI-SVD +\nEuclidiana"]
bars = ax.bar(labs, vals, color=cols)
for br, v in zip(bars, vals):
    ax.text(br.get_x() + br.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
ax.set_ylim(0, max(vals) + 0.1); ax.set_ylabel("Similitud (0 a 1)")
ax.set_title("Abstract 12 vs Abstract 20: similitud segun cada algoritmo")
ax.grid(axis="y", alpha=.3); plt.tight_layout(); plt.savefig("figs/fig_r1_barras.png", dpi=170); plt.close()
# figura: mapa de calor TF-IDF
Mt = np.array(sv.matriz_tfidf_coseno())
fig, ax = plt.subplots(figsize=(6.4, 5.4))
im = ax.imshow(Mt, cmap="Blues", vmin=0, vmax=0.4)
ax.set_xticks(range(n)); ax.set_yticks(range(n))
ax.set_xticklabels(range(1, n + 1), fontsize=8); ax.set_yticklabels(range(1, n + 1), fontsize=8)
plt.colorbar(im, label="Coseno TF-IDF"); ax.set_title("Similitud TF-IDF entre los 20 abstracts (diagonal = 1, recortada)")
plt.tight_layout(); plt.savefig("figs/fig_r1_heatmap.png", dpi=170); plt.close()

# ============================================================= REQUERIMIENTO 2
Sm = sv.matriz_tfidf_coseno()
D = C.distancias_desde_similitud(Sm)
etiq = [f"A{i+1}" for i in range(n)]
r2 = {"metodos": {}}
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
for ax, (m, nom) in zip(axes, (("single", "Single Linkage"), ("complete", "Complete Linkage"), ("average", "Average Linkage"))):
    F = C.aglomerativo(D, m)
    cp = C.cophenetic_corr(D, F)
    sil = {k: C.silhouette(D, C.cortar(F, n, k)) for k in range(2, 9)}
    r2["metodos"][m] = {"nombre": nom, "fusiones": [(int(x), int(y), float(d), int(t)) for x, y, d, t in F],
                        "cpcc": cp, "sil": {str(k): v for k, v in sil.items()}}
    C.dibujar_dendrograma(F, etiq, nom, ax)
    # figura individual
    f2, a2 = plt.subplots(figsize=(7.2, 4.0))
    C.dibujar_dendrograma(F, etiq, f"Dendrograma - {nom}", a2)
    f2.tight_layout(); f2.savefig(f"figs/fig_r2_{m}.png", dpi=170); plt.close(f2)
fig.tight_layout(); fig.savefig("figs/fig_r2_todos.png", dpi=150); plt.close(fig)
for m, info in r2["metodos"].items():
    ks = {int(k): v for k, v in info["sil"].items()}
    kb = max(ks, key=ks.get)
    info["k_mejor"], info["sil_mejor"] = kb, ks[kb]
    F = [tuple(x) for x in info["fusiones"]]
    info["sil_media"] = float(np.mean(list(ks.values())))
# etiquetas del mejor metodo y temas de cada cluster
mejor = max(r2["metodos"], key=lambda m: r2["metodos"][m]["cpcc"] + r2["metodos"][m]["sil_media"])
r2["mejor"] = mejor
F = [tuple(x) for x in r2["metodos"][mejor]["fusiones"]]
def temas(labels, topn=6):
    out = {}
    for lab in sorted(set(labels)):
        miembros = [i for i in range(n) if labels[i] == lab]
        acc = Counter()
        for i in miembros:
            for t, w in S.vector_tfidf(sv.tokens[i], sv.idf).items():
                acc[t] += w / len(miembros)
        out[lab] = {"miembros": [i + 1 for i in miembros], "terminos": [t for t, _ in acc.most_common(topn)]}
    return out
r2["cortes"] = {}
for m in r2["metodos"]:
    Fm = [tuple(x) for x in r2["metodos"][m]["fusiones"]]
    kb = r2["metodos"][m]["k_mejor"]
    lab = C.cortar(Fm, n, kb)
    r2["cortes"][m] = {"k": kb, "etiquetas": lab, "temas": temas(lab)}
fig, ax = plt.subplots(figsize=(6.6, 3.6))
for m, c in (("single", "#c0504d"), ("complete", "#1f4e79"), ("average", "#4f9d5d")):
    ks = sorted(int(k) for k in r2["metodos"][m]["sil"])
    ax.plot(ks, [r2["metodos"][m]["sil"][str(k)] for k in ks], marker="o", label=r2["metodos"][m]["nombre"], color=c)
ax.axhline(0, color="k", lw=.6); ax.set_xlabel("Numero de clusters (k)"); ax.set_ylabel("Silhouette promedio")
ax.set_title("Calidad interna segun el numero de grupos"); ax.legend(); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig("figs/fig_r2_silhouette.png", dpi=170); plt.close(fig)
# tiempo de ejecucion
t0 = time.perf_counter()
for m in ("single", "complete", "average"):
    t1 = time.perf_counter(); C.aglomerativo(D, m); r2["metodos"][m]["tiempo_ms"] = (time.perf_counter() - t1) * 1000
R["r2"] = r2; R["r2"]["D_ejemplo"] = [[round(x, 3) for x in D[i][:6]] for i in range(6)]

# ============================================================= REQUERIMIENTO 3
r3 = {}
texto = "to review and rethink their assessment policies"
patron = "ssess"
r3["texto"], r3["patron"] = texto, patron
r3["lps"] = B.tabla_lps(patron)
r3["malo"] = B.tabla_caracter_malo(patron)
r3["sufijo"] = B.tabla_buen_sufijo(patron)
# --- trazas didacticas
def traza_kmp(t, p):
    lps = B.tabla_lps(p); n_, m = len(t), len(p); j = 0; pasos = []
    for i in range(n_):
        while True:
            ok = t[i] == p[j]
            pasos.append((i, j, t[i], p[j], ok))
            if ok:
                j += 1; break
            if j == 0: break
            j = lps[j - 1]
        if j == m:
            pasos.append((i, m, "", "", "MATCH en %d" % (i - m + 1))); j = lps[j - 1]
    return pasos
def traza_bm(t, p):
    malo = B.tabla_caracter_malo(p); suf = B.tabla_buen_sufijo(p); n_, m = len(t), len(p); s = 0; pasos = []
    while s <= n_ - m:
        j = m - 1; comps = 0
        while j >= 0 and p[j] == t[s + j]:
            j -= 1; comps += 1
        if j < 0:
            pasos.append({"s": s, "comp": comps, "j": None, "mal": None, "bm": None, "bs": suf[0], "desp": suf[0], "match": True}); s += suf[0]
        else:
            comps += 1
            bm = j - malo.get(t[s + j], -1); bs = suf[j + 1]
            pasos.append({"s": s, "comp": comps, "j": j, "mal": t[s + j], "bm": bm, "bs": bs, "desp": max(bm, bs), "match": False}); s += max(bm, bs)
    return pasos
r3["traza_kmp"] = traza_kmp(texto, patron)
r3["traza_bm"] = traza_bm(texto, patron)
r3["res_kmp"] = B.kmp(texto, patron); r3["res_bm"] = B.boyer_moore(texto, patron)
q, base = 101, 256
rk = B.rabin_karp(texto, patron, base, q)
r3["res_rk"] = rk
m = len(patron); hp = 0
r3["pot"] = pow(base, m - 1, q)
ventanas = []
ht = 0
for i in range(m): ht = (base * ht + ord(texto[i])) % q
hp = 0
for ch in patron: hp = (base * hp + ord(ch)) % q
for i in range(len(texto) - m + 1):
    w = texto[i:i + m]
    ventanas.append({"i": i, "w": w, "h": ht, "igual_hash": ht == hp, "igual_texto": w == patron})
    if i < len(texto) - m: ht = (base * (ht - ord(texto[i]) * r3["pot"]) + ord(texto[i + m])) % q
r3["rk_ventanas"] = ventanas; r3["rk_hp"] = hp; r3["rk_q"] = q; r3["rk_base"] = base
r3["rk_calc_patron"] = [(ch, ord(ch)) for ch in patron]
r3["rk_colisiones_q101"] = rk["colisiones"]
r3["rk_ejemplo_roll"] = None
# busqueda real en los 20 abstracts
res = {}
for pat in ["assessment", "ssess", "artificial intelligence", "generative", "higher education"]:
    fila = {"patron": pat}
    for nm, f in B.ALGORITMOS.items():
        tot_pos = tot_comp = 0; t0 = time.perf_counter(); por_doc = []
        for d in sv.docs:
            r = f(d["abstract"].lower(), pat.lower())
            tot_pos += len(r["posiciones"]); tot_comp += r["comparaciones"]; por_doc.append(len(r["posiciones"]))
        fila[nm] = {"ocurrencias": tot_pos, "comparaciones": tot_comp, "ms": (time.perf_counter() - t0) * 1000, "por_doc": por_doc}
    res[pat] = fila
r3["abstracts"] = res
# benchmark en texto grande
completo = ""
for f in sorted(glob.glob("full/*.txt"), key=lambda x: int(os.path.basename(x).split(".")[0])):
    completo += open(f, encoding="utf8", errors="ignore").read().lower() + " "
completo = " ".join(completo.split())
r3["n_texto_completo"] = len(completo)
def medir(f, t, p, rep=2):
    best = 1e9; r = None
    for _ in range(rep):
        t0 = time.perf_counter(); r = f(t, p); best = min(best, time.perf_counter() - t0)
    return best * 1000, r
patrones = ["ssess", "assessment", "artificial intelligence", "generative artificial intelligence in higher education", "teaching and learning practice in higher education institutions around the world"]
bench = []
algs = dict(B.ALGORITMOS); algs["Fuerza bruta"] = B.fuerza_bruta
for p in patrones:
    fila = {"patron": p, "m": len(p)}
    for nm, f in algs.items():
        ms, r = medir(f, completo, p)
        fila[nm] = {"ms": ms, "comp": r["comparaciones"], "occ": len(r["posiciones"]), "extra": r.get("colisiones")}
    bench.append(fila)
r3["bench"] = bench
# crecimiento con n
crec = {nm: [] for nm in algs}; tams = [100000, 200000, 400000, 800000, len(completo)]
for tam in tams:
    for nm, f in algs.items():
        ms, _ = medir(f, completo[:tam], "generative artificial intelligence", rep=1)
        crec[nm].append(ms)
r3["crec"] = {"tams": tams, "ms": crec}
fig, ax = plt.subplots(figsize=(6.6, 3.8))
for nm, c in zip(algs, ["#1f4e79", "#c0504d", "#4f9d5d", "#999999"]):
    ax.plot([t / 1000 for t in tams], crec[nm], marker="o", label=nm, color=c)
ax.set_xlabel("Tamano del texto (miles de caracteres)"); ax.set_ylabel("Tiempo (ms)")
ax.set_title("Tiempo de busqueda vs tamano del texto"); ax.legend(); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig("figs/fig_r3_crecimiento.png", dpi=170); plt.close(fig)
# comparaciones por longitud de patron
fig, ax = plt.subplots(figsize=(6.6, 3.8)); w = 0.2
for k, (nm, c) in enumerate(zip(algs, ["#1f4e79", "#c0504d", "#4f9d5d", "#999999"])):
    ax.bar(np.arange(len(patrones)) + k * w, [f[nm]["comp"] / 1e6 for f in bench], w, label=nm, color=c)
ax.set_xticks(np.arange(len(patrones)) + 1.5 * w); ax.set_xticklabels([f"m={f['m']}" for f in bench])
ax.set_ylabel("Comparaciones de caracteres (millones)"); ax.set_title("Comparaciones segun la longitud del patron")
ax.legend(); ax.grid(axis="y", alpha=.3); fig.tight_layout(); fig.savefig("figs/fig_r3_comparaciones.png", dpi=170); plt.close(fig)
# peor caso / adversarial
adv = {}
for nombre, (t, p) in {"Texto 'a' x 100000, patron 'a'x49+'b'": ("a" * 100000, "a" * 49 + "b"),
                       "Texto 'a' x 100000, patron 'a' x 50": ("a" * 100000, "a" * 50),
                       "Texto aleatorio (4 letras) x 100000, patron 20": ("".join(np.random.default_rng(1).choice(list("acgt"), 100000)), "acgtacgtacgtacgtacgt")}.items():
    adv[nombre] = {}
    for nm, f in algs.items():
        ms, r = medir(f, t, p, rep=1); adv[nombre][nm] = {"ms": ms, "comp": r["comparaciones"], "occ": len(r["posiciones"])}
r3["adv"] = adv
# preprocesamiento vs busqueda
pre = {}
for m_ in (10, 100, 1000, 5000):
    p = completo[1000:1000 + m_]
    fila = {}
    def tiempo(fn, rep=20):
        t0 = time.perf_counter()
        for _ in range(rep): fn()
        return (time.perf_counter() - t0) / rep * 1000
    fila["KMP"] = tiempo(lambda: B.tabla_lps(p))
    fila["Boyer-Moore"] = tiempo(lambda: (B.tabla_caracter_malo(p), B.tabla_buen_sufijo(p)))
    def rkpre():
        h = 0
        for ch in p: h = (256 * h + ord(ch)) % 1000000007
    fila["Rabin-Karp"] = tiempo(rkpre)
    fila["busq_ms"] = {nm: medir(f, completo[:200000], p, rep=1)[0] for nm, f in B.ALGORITMOS.items()}
    pre[str(m_)] = fila
r3["pre"] = pre
R["r3"] = r3
json.dump(R, open("resultados.json", "w"), ensure_ascii=False, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating,)) else str(o))
print("OK")
