
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Carpeta donde guardaremos los modelos entrenados
CARPETA_MODELOS = Path("trained_models")
CARPETA_MODELOS.mkdir(exist_ok=True)


# Columnas que necesitamos del dataset NASA POWER
COLUMNAS_NECESARIAS = [
    "YEAR",
    "MO",
    "DY",
    "ALLSKY_SFC_SW_DWN"
]

# Preparamos los datos para el entrenamiento:
# Esta función verifica que el archivo tenga las columnas necesarias,
# convierte los valores a números y elimina los registros vacíos
# o que contienen -999. Después crea una fecha con el año, mes y día,
# calcula dos variables (seno y coseno) para representar la época del año
# y separa los datos de entrada (X) de la irradiación solar (y).
# Preparamos los datos antes de entrenar
def preparar_datos(df):
    df = df.copy()
    # Verificamos que existan todas las columnas necesarias
    for columna in COLUMNAS_NECESARIAS:
        if columna not in df.columns:
            raise ValueError(f"Falta la columna: {columna}")
    for columna in COLUMNAS_NECESARIAS:  # Convertimos los datos a valores numéricos
        df[columna] = pd.to_numeric(df[columna], errors="coerce")
    # Eliminamos los valores faltantes de las columnas necesarias
    df = df.dropna(subset=COLUMNAS_NECESARIAS)
    # NASA POWER utiliza -999 para representar datos faltantes
    df = df[df["ALLSKY_SFC_SW_DWN"] != -999]
    # Creamos una columna con la fecha completa
    df["FECHA"] = pd.to_datetime(dict(year=df["YEAR"], month=df["MO"], day=df["DY"]), errors="coerce")
    # Eliminamos fechas inválidas
    df = df.dropna(subset=["FECHA"])
    # Ordenamos los datos cronológicamente
    df = df.sort_values("FECHA").reset_index(drop=True)
    # Obtenemos el día del año (1 a 366)
    dia_años = df["FECHA"].dt.dayofyear
    # Convertimos el día del año en variables cíclicas
    # Esto permite representar que diciembre y enero están cerca
    df["dia_sin"] = np.sin(2 * np.pi * dia_años / 365.25)
    df["dia_cos"] = np.cos(2 * np.pi * dia_años / 365.25)
    # Variables que utilizará el modelo
    X = df[["dia_sin", "dia_cos"]]
    # Valor que queremos predecir: radiación solar diaria
    y = df["ALLSKY_SFC_SW_DWN"]
    # La radiación solar no debe ser negativa
    datos_validos = y >= 0
    X = X[datos_validos]
    y = y[datos_validos]

    return X, y


# Entrenamos el modelo
def entrenar_modelo(df, dataset_id):

    # Preparamos los datos
    X, y = preparar_datos(df)

    # Verificamos que tengamos suficientes registros
    if len(X) < 30:
        raise ValueError(
            "Se necesitan al menos 30 registros válidos para entrenar."
        )

    # Dividimos los datos respetando el orden cronológico
    punto_division = int(len(X) * 0.80)

    X_train = X.iloc[:punto_division]
    X_test = X.iloc[punto_division:]

    y_train = y.iloc[:punto_division]
    y_test = y.iloc[punto_division:]

    # Creamos el árbol de decisión
    modelo = DecisionTreeRegressor(
        max_depth=5,
        min_samples_leaf=20,
        random_state=42
    )

    # Entrenamos con el 80% inicial de los datos
    modelo.fit(X_train, y_train)

    # Realizamos predicciones con el 20% restante
    predicciones = modelo.predict(X_test)

    # Calculamos las métricas de evaluación
    mae = mean_absolute_error(y_test, predicciones)

    rmse = np.sqrt(
        mean_squared_error(y_test, predicciones)
    )

    r2 = r2_score(y_test, predicciones)

    # Entrenamos nuevamente con todos los datos disponibles
    modelo.fit(X, y)

    # Guardamos el modelo correspondiente al dataset
    ruta_modelo = CARPETA_MODELOS / f"modelo_solar_{dataset_id}.pkl"

    joblib.dump(modelo, ruta_modelo)

    # Devolvemos los resultados del entrenamiento
    return {
        "mensaje": "Modelo entrenado correctamente",
        "dataset_id": dataset_id,
        "registros_utilizados": len(X),
        "registros_entrenamiento": len(X_train),
        "registros_prueba": len(X_test),
        "mae": round(float(mae), 4),
        "rmse": round(float(rmse), 4),
        "r2": round(float(r2), 4),
        "modelo_guardado": str(ruta_modelo),
        "Dataset": df.head(5).to_dict()

    }


# Realizamos una predicción para una fecha específica
def predecir_irradiacion(dataset_id, fecha):

    # Buscamos el modelo correspondiente al dataset
    ruta_modelo = CARPETA_MODELOS / f"modelo_solar_{dataset_id}.pkl"

    if not ruta_modelo.exists():
        raise FileNotFoundError(
            "Primero debes entrenar el modelo para este dataset."
        )

    # Cargamos el modelo entrenado
    modelo = joblib.load(ruta_modelo)

    # Convertimos la fecha recibida
    fecha = pd.to_datetime(fecha)

    # Obtenemos el día del año
    dia_años = fecha.dayofyear

    # Creamos las mismas variables utilizadas durante el entrenamiento
    datos = pd.DataFrame([{
        "dia_sin": np.sin(2 * np.pi * dia_años / 365.25),
        "dia_cos": np.cos(2 * np.pi * dia_años / 365.25)
    }])

    # Realizamos la predicción
    resultado = modelo.predict(datos)[0]

    # Devolvemos la radiación solar estimada
    return round(float(resultado), 4)
