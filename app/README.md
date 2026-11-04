# Analisis de algoritmos en bibliometria (IA generativa en educacion)

Aplicacion web (Flask) con los tres requerimientos del proyecto. Todos los algoritmos estan escritos desde cero.

## Estructura
- `app.py` – servidor web y API (`/api/similitud`, `/api/clustering`, `/api/busqueda`).
- `servicio.py` – une el corpus con los algoritmos.
- `algoritmos/` – `similitud.py`, `embeddings.py`, `clustering.py`, `busqueda.py`, `preprocesamiento.py`.
- `data/corpus.json` – titulo, autores, anio y abstract de los 20 articulos; `data/*.npz` – embeddings entrenados.
- `entrenar_embeddings.py` – reentrena los embeddings a partir de los textos completos (opcional).
- `tests/test_algoritmos.py` – pruebas automaticas (en la carpeta superior).

## Ejecutar en local
```
pip install -r requirements.txt
python app.py            # abre http://localhost:5000
```

## Despliegue
- **Render / Railway / Heroku:** subir esta carpeta a un repositorio de GitHub, crear un "Web Service" con
  `pip install -r requirements.txt` como build y `gunicorn app:app` como start (el `Procfile` ya lo define).
- **Docker:** `docker build -t bibliometria .` y `docker run -p 8080:8080 bibliometria`.
- Comprobacion de salud: `GET /api/salud` devuelve `{"estado":"ok","documentos":20}`.
