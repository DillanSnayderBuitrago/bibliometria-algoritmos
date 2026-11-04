"""Aplicacion web (Flask) - Analisis de algoritmos en bibliometria.
Ejecutar local:  python app.py     (http://localhost:5000)
Produccion:      gunicorn app:app
"""
import base64, html, io, time
from flask import Flask, jsonify, render_template, request
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from servicio import Servicio, NOMBRES_SIM, TIPO
from algoritmos import clustering as C, busqueda as B

app = Flask(__name__)
sv = Servicio()
_cache_cluster = {}

def figura_a_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

@app.route("/")
def inicio():
    return render_template("index.html")

@app.route("/api/salud")
def salud():
    return jsonify(estado="ok", documentos=len(sv.docs))

@app.route("/api/corpus")
def corpus():
    return jsonify([{k: d[k] for k in ("id", "title", "authors", "year", "abstract")} for d in sv.docs])

# ---------------------------------------------------------------- Requerimiento 1
@app.route("/api/similitud", methods=["POST"])
def similitud():
    ids = request.get_json(force=True).get("ids", [])
    ids = sorted(set(int(i) for i in ids))
    if len(ids) < 2 or any(i < 1 or i > len(sv.docs) for i in ids):
        return jsonify(error="Seleccione al menos dos articulos validos (1-20)."), 400
    pares = []
    for x in range(len(ids)):
        for y in range(x + 1, len(ids)):
            i, j = ids[x] - 1, ids[y] - 1
            fila = {"a": ids[x], "b": ids[y], "valores": {}, "ms": {}}
            for nm in NOMBRES_SIM:
                t0 = time.perf_counter()
                v = float(sv.sim(nm, i, j))
                fila["ms"][nm] = round((time.perf_counter() - t0) * 1000, 3)
                fila["valores"][nm] = round(v, 4)
            pares.append(fila)
    return jsonify(algoritmos=[{"nombre": n, "tipo": TIPO[n]} for n in NOMBRES_SIM], pares=pares,
                   titulos={i: sv.docs[i - 1]["title"] for i in ids})

# ---------------------------------------------------------------- Requerimiento 2
@app.route("/api/clustering", methods=["POST"])
def clustering():
    if "res" in _cache_cluster:
        return jsonify(_cache_cluster["res"])
    n = len(sv.docs)
    D = C.distancias_desde_similitud(sv.matriz_tfidf_coseno())
    etiquetas = [f"A{i+1}" for i in range(n)]
    salida = []
    for m, nombre in (("single", "Single Linkage"), ("complete", "Complete Linkage"), ("average", "Average Linkage")):
        t0 = time.perf_counter()
        F = C.aglomerativo(D, m)
        ms = (time.perf_counter() - t0) * 1000
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        C.dibujar_dendrograma(F, etiquetas, f"Dendrograma - {nombre}", ax)
        sil = {k: round(C.silhouette(D, C.cortar(F, n, k)), 4) for k in range(2, 9)}
        salida.append({"metodo": m, "nombre": nombre, "imagen": figura_a_base64(fig), "cpcc": round(C.cophenetic_corr(D, F), 4),
                       "silhouette": sil, "sil_medio": round(sum(sil.values()) / len(sil), 4), "ms": round(ms, 2),
                       "grupos_k4": C.cortar(F, n, 4)})
    mejor = max(salida, key=lambda s: s["cpcc"] + s["sil_medio"])["nombre"]
    res = {"resultados": salida, "mejor": mejor,
           "titulos": {f"A{d['id']}": d["title"] for d in sv.docs}}
    _cache_cluster["res"] = res
    return jsonify(res)

# ---------------------------------------------------------------- Requerimiento 3
def fragmento(texto, pos, m, ancho=45):
    ini, fin = max(0, pos - ancho), min(len(texto), pos + m + ancho)
    return (("..." if ini > 0 else "") + html.escape(texto[ini:pos]) + "<mark>" + html.escape(texto[pos:pos + m]) + "</mark>"
            + html.escape(texto[pos + m:fin]) + ("..." if fin < len(texto) else ""))

@app.route("/api/busqueda", methods=["POST"])
def busqueda():
    datos = request.get_json(force=True)
    patron = (datos.get("patron") or "")
    campo = datos.get("campo", "abstract")
    ignorar = bool(datos.get("ignorar_mayusculas", True))
    alg = datos.get("algoritmo", "Todos")
    if not patron.strip():
        return jsonify(error="Escriba un patron de busqueda."), 400
    if campo not in ("abstract", "title", "authors"):
        return jsonify(error="Campo no valido."), 400
    algs = list(B.ALGORITMOS) if alg == "Todos" else [alg]
    if any(a not in B.ALGORITMOS for a in algs):
        return jsonify(error="Algoritmo no valido."), 400
    p = patron.lower() if ignorar else patron
    resumen = {}
    docs_res = {}
    for a in algs:
        f = B.ALGORITMOS[a]
        t_total = 0.0; comp = 0; occ = 0
        for d in sv.docs:
            texto = d[campo]
            t = texto.lower() if ignorar else texto
            t0 = time.perf_counter()
            r = f(t, p)
            t_total += time.perf_counter() - t0
            comp += r["comparaciones"]; occ += len(r["posiciones"])
            if r["posiciones"]:
                docs_res.setdefault(d["id"], {"id": d["id"], "title": d["title"], "authors": d["authors"], "posiciones": r["posiciones"],
                                              "fragmentos": [fragmento(texto, q, len(p)) for q in r["posiciones"][:5]]})
        extra = {}
        if a == "Rabin-Karp":
            extra = {"colisiones": sum(f(((d[campo].lower()) if ignorar else d[campo]), p)["colisiones"] for d in sv.docs)}
        resumen[a] = {"ocurrencias": occ, "comparaciones": comp, "ms": round(t_total * 1000, 3), **extra}
    # verificacion cruzada: los algoritmos deben dar el mismo numero de ocurrencias
    ocurr = {v["ocurrencias"] for v in resumen.values()}
    return jsonify(patron=patron, resumen=resumen, consistente=len(ocurr) == 1,
                   documentos=sorted(docs_res.values(), key=lambda x: x["id"]))

if __name__ == "__main__":
    import os
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
