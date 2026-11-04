import json, os
RUTA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "corpus.json")

def cargar_corpus():
    with open(RUTA, encoding="utf8") as f:
        return json.load(f)
