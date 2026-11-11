from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session
from io import BytesIO
from pathlib import Path
import uuid
import pandas as pd
from app.db.dependencies import get_db
from app.schemas import DatasetResponse
from app.models.Dataset import Dataset
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


