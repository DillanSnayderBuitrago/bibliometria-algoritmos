"""Preprocesamiento de texto: minusculas, tokens, stopwords y normalizacion simple."""
import re

STOPWORDS = set("""a about above after again against all also am an and any are as at be because been before being
below between both but by can could did do does doing down during each few for from further had has have having he her
here hers him his how i if in into is it its itself just me more most my no nor not of off on once only or other our
ours out over own same she should so some such than that the their theirs them then there these they this those through
to too under until up very was we were what when where which while who whom why will with would you your et al e.g i.e
via may might must shall within without among across per us thus however therefore moreover although whereas whether
based using used use one two three four five six first second third new """.split())


def normalizar_palabra(w):
    """Normalizacion ligera de plurales: ies->y, ches/shes/xes->sin 'es', y se quita la 's' final."""
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 5 and w.endswith(("ches", "shes", "xes", "zes")):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        return w[:-1]
    return w


def tokenizar(texto):
    """Devuelve la lista de palabras en minuscula (solo letras, guiones internos se separan)."""
    return re.findall(r"[a-z]+", texto.lower())


def preprocesar(texto, quitar_stopwords=True, normalizar=True):
    """Pipeline completo: tokens -> sin stopwords -> normalizacion."""
    salida = []
    for t in tokenizar(texto):
        if len(t) < 2:
            continue
        if quitar_stopwords and t in STOPWORDS:
            continue
        if normalizar:
            t = normalizar_palabra(t)
        salida.append(t)
    return salida
