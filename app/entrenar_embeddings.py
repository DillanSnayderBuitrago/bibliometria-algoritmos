"""Entrena los dos modelos de embeddings y los guarda en data/. Uso: python entrenar_embeddings.py <carpeta_con_textos_completos>"""
import sys, glob, os, time, json
sys.path.insert(0, os.path.dirname(__file__))
from algoritmos.preprocesamiento import preprocesar
from algoritmos import embeddings as E

carpeta = sys.argv[1] if len(sys.argv) > 1 else "../full"
textos = []
for f in sorted(glob.glob(os.path.join(carpeta, "*.txt"))):
    raw = open(f, encoding="utf8", errors="ignore").read()
    # se descarta la lista de referencias (ultima aparicion de 'References' en la segunda mitad)
    corte = max(raw.rfind("\nReferences"), raw.rfind("\nREFERENCES"), raw.rfind("\nBibliography"))
    if corte > len(raw) * 0.5:
        raw = raw[:corte]
    textos.append(preprocesar(raw))
print("documentos:", len(textos), "tokens:", sum(map(len, textos)))
t0 = time.time()
pal, W, perd = E.entrenar_sgns(textos, dim=50, epocas=6)
E.guardar("w2v_sgns", pal, W)
print("SGNS vocab", len(pal), "perdida por epoca", [round(p, 3) for p in perd], f"{time.time()-t0:.0f}s")
t0 = time.time()
pal2, W2 = E.entrenar_ppmi_svd(textos, dim=50)
E.guardar("ppmi_svd", pal2, W2)
print("PPMI-SVD vocab", len(pal2), f"{time.time()-t0:.0f}s")
idx = {p: i for i, p in enumerate(pal)}; idx2 = {p: i for i, p in enumerate(pal2)}
info = {"tokens": sum(map(len, textos)), "vocab_sgns": len(pal), "vocab_ppmi": len(pal2), "perdida": perd}
for w in ["assessment", "chatgpt", "teacher", "review", "generative"]:
    print(w, "SGNS:", [(a, round(b, 2)) for a, b in E.vecinos(w, pal, idx, W, 5)])
    print(w, "PPMI:", [(a, round(b, 2)) for a, b in E.vecinos(w, pal2, idx2, W2, 5)])
json.dump(info, open("data/embeddings_info.json", "w"))
