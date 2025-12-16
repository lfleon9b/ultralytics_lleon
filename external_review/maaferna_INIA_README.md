# Proyecto de Entrenamiento de YOLOv8

Este proyecto está diseñado para entrenar modelos YOLOv8 en diferentes versiones utilizando ClearML para el seguimiento de experimentos y la gestión de resultados. 

## Descripción

Este repositorio contiene un script para entrenar modelos de la serie YOLOv8 (`n`, `s`, `m`, `l`, `x`) con parámetros ajustables como el tamaño del lote, el tamaño de la imagen, el número de ejecuciones y el número de épocas. Los resultados del entrenamiento se registran y se informan a ClearML para su seguimiento y análisis.

## Requisitos

Antes de ejecutar el proyecto, asegúrate de tener los siguientes requisitos instalados:

- Python 3.6 o superior
- Conda o pip para la gestión de paquetes
- Dependencias del proyecto (especificadas en `requirements.txt`)

## Instalación

1. Clona el repositorio:
    ```bash
    git clone <URL_DEL_REPOSITORIO>
    cd <NOMBRE_DEL_REPOSITORIO>
    ```

Crea el entorno virtual desde el archivo environment.yml:
    ```bash
    conda env create -f environment.yml
    conda activate yolov8_custom
    ```
Esto instalará todas las dependencias necesarias, incluyendo las manejadas por Pip, si están listadas en la sección pip del archivo environment.yml.

Alternativa con Pipenv

Si prefieres usar Pipenv en lugar de Conda, puedes generar un archivo requirements.txt desde el archivo environment.yml y usarlo para instalar las dependencias en un entorno virtual administrado por Pipenv.

Genera un archivo requirements.txt:
```bash
conda env export --no-builds | grep -A 1000 "^dependencies:" | grep -v "^prefix:" > requirements.txt
```

Instala las dependencias usando Pipenv:
```bash
pipenv install -r requirements.txt

```

3. Instala las dependencias:
    ```bash
    pip install -r requirements.txt
    ```

4. Configura tus credenciales de ClearML en un archivo `config.py`. Asegúrate de definir las siguientes variables:
    ```python
    CLEARML_API_HOST = 'https://api.clear.ml'
    CLEARML_WEB_HOST = 'https://app.clear.ml'
    CLEARML_FILES_HOST = 'https://files.clear.ml'
    CLEARML_KEY_ORIGINAL = '<TU_API_KEY_ORIGINAL>'
    CLEARML_SECRET_ORIGINAL = '<TU_API_SECRET_ORIGINAL>'
    CLEARML_KEY_PERSONAL = '<TU_API_KEY_PERSONAL>'
    CLEARML_SECRET_PERSONAL = '<TU_API_SECRET_PERSONAL>'
    ```

## Uso

Modifica los parámetros en el script main.py según tus necesidades, como el tamaño de la imagen, el número de épocas, etc.

Ejecuta el script main.py para iniciar el entrenamiento o realizar predicciones:

1. Modifica los parámetros en el script `main.py` según tus necesidades. Los parámetros ajustables incluyen:
    - `model_version`: Versión del modelo YOLOv8 (`n`, `s`, `m`, `l`, `x`)
    - `batch_size`: Tamaño del lote para el entrenamiento
    - `imgsz`: Tamaño de la imagen para el entrenamiento
    - `num_runs`: Número de ejecuciones
    - `base_seed`: Semilla base para la generación de números aleatorios
    - `epochs`: Número de épocas de entrenamiento
    - `device`: GPU a utilizar (especificar el índice de la GPU)

2. Ejecuta el script `main.py`:
    ```bash
    python scripts/main.py
    ```

## Resultados

Los resultados del entrenamiento se registran en ClearML y se almacenan en un archivo de log en el directorio `outputs/logs`. Puedes ver los resultados y métricas del entrenamiento en el panel de ClearML.

## Contribuciones

Las contribuciones son bienvenidas. Si deseas contribuir al proyecto, por favor, sigue estos pasos:

1. Haz un fork del repositorio.
2. Crea una nueva rama para tus cambios.
3. Realiza un pull request con una descripción de tus cambios.

## Licencia

Este proyecto está bajo la Licencia MIT. Consulta el archivo `LICENSE` para más detalles.

---

Para cualquier pregunta o problema, por favor, contacta al equipo de desarrollo a través de los canales disponibles en el repositorio.


(base) C:\Users\Usuario>cd yolov8_project/yolov8_project_test

(base) C:\Users\Usuario\yolov8_project\yolov8_project_test>conda activate yolov8_custom

(yolov8_custom) C:\Users\Usuario\yolov8_project\yolov8_project_test>cd scripts

(yolov8_custom) C:\Users\Usuario\yolov8_project\yolov8_project_test\scripts>