# Data Analyzer

Una API sencilla hecha con FastAPI y Pandas para subir archivos `.csv` y sacar estadísticas y análisis estadísticos

Además de leer CSV normales, detecta y limpia automáticamente los encabezados de archivos descargados de NASA POWER.

## Herramientas
* Python y FastAPI
* Pandas (para leer y analizar los datos)
* PostgreSQL, SQLAlchemy y Alembic (para guardar el registro de los archivos)
* Docker

## ¿Qué puedes hacer con la API?
* `POST /datasets/` — Sube un archivo `.csv` y lo guarda en la base de datos.
* `GET /datasets` — Muestra la lista de todos los archivos que has subido.
* `GET /datasets/{id}` — Busca un archivo en específico por su ID.
* `GET /datasets/{id}/statics` — Te da las estadísticas básicas (promedios, mínimos, máximos, etc.).
* `GET /datasets/{id}/missing-values` — Revisa si hay celdas vacías o datos faltantes por columna.
* `GET /datasets/{id}/info` — Muestra cuántas filas y columnas tiene, sus nombres y tipos de datos.
* `GET /datasets/{id}/analysis` — Separa columnas numéricas de texto y saca las correlaciones entre números.
