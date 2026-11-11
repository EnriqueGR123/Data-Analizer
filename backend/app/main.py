
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session
from io import BytesIO
from pathlib import Path
import uuid
import pandas as pd
import numpy as np
from app.db.dependencies import get_db
from app.schemas import DatasetResponse, PredictionRequest
from app.models.Dataset import Dataset
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from app.services.machine_learning import preparar_datos

app = FastAPI(title="Aura Jalisco")


# Carpetas donde guardaremos los archivos y modelos
UPLOAD_DIR = Path("uploads")
MODEL_DIR = Path("trained_models")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# Lee archivos CSV normales y archivos con encabezado de NASA POWER
def leer_csv(content):
    lineas = content.decode("utf-8-sig").splitlines()
    for numero, linea in enumerate(lineas):
        if linea.strip().startswith("YEAR,MO,DY"):
            return pd.read_csv(BytesIO(content), skiprows=numero)
    return pd.read_csv(BytesIO(content))


# Busca un dataset y carga su archivo CSV
def obtener_dataset_y_archivo(id, db):
    dataset = db.query(Dataset).filter(Dataset.id == id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    ruta = Path(dataset.file_path)
    if not ruta.exists():
        raise HTTPException(status_code=404, detail="CSV file not found")
    try:
        df = leer_csv(ruta.read_bytes())
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError):
        raise HTTPException(status_code=400, detail="Invalid CSV file")
    return dataset, df




@app.get("/inicio")
def inicio():
    return {"mensaje": "Aura Jalisco API funcionando"}


# Subir un archivo CSV
@app.post("/datasets/", response_model=DatasetResponse)
async def upload_file(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Debe ser un archivo CSV")
    content = await file.read()
    try:
        df = leer_csv(content)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError):
        raise HTTPException(status_code=400, detail="Invalid CSV file")

    file_id = uuid.uuid4()
    filename = Path(file.filename).name
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    file_path = UPLOAD_DIR / f"{file_id}_{filename}"
    try:
        file_path.write_bytes(content)
        new_file = Dataset(filename=filename, file_path=str(file_path), rows=df.shape[0], columns=df.shape[1])
        db.add(new_file)
        db.commit()
        db.refresh(new_file)
    except Exception as e:
        print("ERROR AL GUARDAR:", repr(e))
        db.rollback()
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="No se pudo guardar el archivo")
    return new_file


# Consultar todos los datasets guardados
@app.get("/datasets", response_model=list[DatasetResponse])
def get_datasets(db: Session = Depends(get_db)):
    datasets = db.query(Dataset).all()
    return datasets


# Consultar un dataset por su ID
@app.get("/datasets/{id}", response_model=DatasetResponse)
def get_dataset_by_id(id: int, db: Session = Depends(get_db)):
    dataset = db.query(Dataset).filter(Dataset.id == id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return dataset


# Obtener estadísticas del archivo
@app.get("/datasets/{id}/statics")
def get_statics(id: int, db: Session = Depends(get_db)):
    dataset, df = obtener_dataset_y_archivo(id, db)
    return {"rows": df.shape[0], "columns": df.shape[1], "statistics": df.describe().to_dict()}


# Obtener valores faltantes
@app.get("/datasets/{id}/missing-values")
def missing_values(id: int, db: Session = Depends(get_db)):
    dataset, df = obtener_dataset_y_archivo(id, db)
    return df.isnull().sum().to_dict()


# Obtener información del archivo
@app.get("/datasets/{id}/info")
def dataset_info(id: int, db: Session = Depends(get_db)):
    dataset, df = obtener_dataset_y_archivo(id, db)
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": df.columns.to_list(),
        "data_types": df.dtypes.astype(str).to_dict()
    }


# Analizar las columnas y las correlaciones del archivo
@app.get("/datasets/{id}/analysis")
def get_analysis(id: int, db: Session = Depends(get_db)):
    dataset, df = obtener_dataset_y_archivo(id, db)

    categorical = df.select_dtypes(exclude="number")
    numeric = df.select_dtypes(include="number")

    return {
        "dataset": dataset.filename,
        "numeric_columns": numeric.columns.to_list(),
        "numeric_columns_count": len(numeric.columns),
        "categorical_columns": categorical.columns.to_list(),
        "categorical_columns_count": len(categorical.columns),
        "numeric_summary": numeric.describe().to_dict(),
        "correlaciones": numeric.corr().to_dict()
    }


# Entrenar el árbol de decisión con un dataset
# Entrenamos el modelo:
# Esta función utiliza los datos preparados para entrenar un árbol de decisión.
# Primero divide los datos en 80% para entrenamiento y 20% para pruebas.
# Después entrena el modelo y evalúa sus resultados mediante las métricas
# MAE, RMSE y R2. Finalmente, vuelve a entrenarlo con todos los registros
# y guarda el modelo para utilizarlo posteriormente en las predicciones.
@app.post("/datasets/{id}/train")
def train_dataset(id: int, db: Session = Depends(get_db)):
    dataset, df = obtener_dataset_y_archivo(id, db)
    try:
        X, y = preparar_datos(df)
        if len(X) < 30:
            raise ValueError("Se necesitan al menos 30 registros válidos")
        # Dividimos los datos: 80% entrenamiento y 20% prueba
        punto = int(len(X) * 0.80)
        X_train, X_test = X.iloc[:punto], X.iloc[punto:]
        y_train, y_test = y.iloc[:punto], y.iloc[punto:]
        modelo = DecisionTreeRegressor(max_depth=5, min_samples_leaf=20, random_state=42)
        modelo.fit(X_train, y_train)
        # Evaluamos el modelo
        predicciones = modelo.predict(X_test)
        mae = mean_absolute_error(y_test, predicciones)
        rmse = np.sqrt(mean_squared_error(y_test, predicciones))
        r2 = r2_score(y_test, predicciones)
        # Entrenamos con todos los datos y guardamos el modelo
        modelo.fit(X, y)
        ruta_modelo = MODEL_DIR / f"modelo_solar_{id}.pkl"
        joblib.dump(modelo, ruta_modelo)
        resultado = {
            "registros_utilizados": len(X),
            "mae": round(float(mae), 4),
            "rmse": round(float(rmse), 4),
            "r2": round(float(r2), 4),
        
        }
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    return {"mensaje": "Modelo entrenado correctamente", "resultado": resultado}


# Realizar una predicción de irradiación solar
# Realizamos una predicción de irradiación solar:
# Esta función recibe una fecha y busca el modelo que fue entrenado anteriormente
# Comprueba que el modelo exista, lo carga y calcula las variables de seno y coseno
# correspondientes a la fecha solicitada. Después introduce esos valores al modelo
# para estimar la irradiación solar diaria y devuelve el resultado en kWh/m2/dia
@app.post("/datasets/{id}/predict")
def predict_dataset(id: int, datos: PredictionRequest, db: Session = Depends(get_db)):
    dataset, df = obtener_dataset_y_archivo(id, db)
    dataset = db.query(Dataset).filter(Dataset.id == id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    ruta_modelo = MODEL_DIR / f"modelo_solar_{id}.pkl"
    if not ruta_modelo.exists():
        raise HTTPException(status_code=404, detail="Primero debes entrenar el modelo")
    try:
        modelo = joblib.load(ruta_modelo)
        fecha = pd.to_datetime(datos.fecha)
        dia = fecha.dayofyear
        entrada = pd.DataFrame([{"dia_sin": np.sin(2 * np.pi * dia / 365.25),"dia_cos": np.cos(2 * np.pi * dia / 365.25)}])
        resultado = modelo.predict(entrada)[0]
    except (ValueError, OSError) as error:
        raise HTTPException(status_code=400, detail=str(error))
    return {
        "dataset": dataset.filename,
        "fecha": datos.fecha.isoformat(),
        "irradiacion_predicha": round(float(resultado), 4),
        "unidad": "kWh/m2/dia",
        "rows": df.shape[0],
    }
