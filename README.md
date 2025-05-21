# Detector de Mascarillas con EfficientDet-D1 y TensorFlow

Este proyecto implementa un detector de mascarillas en tiempo real utilizando una cámara web. Se basa en el modelo EfficientDet-D1 afinado del repositorio [coolmunzi/face_mask_detector](https://github.com/coolmunzi/face_mask_detector), que utiliza la API de Detección de Objetos de TensorFlow 2.

## Características

*   Detección en tiempo real de:
    *   Personas **con mascarilla** (etiqueta `with_mask`).
    *   Personas **sin mascarilla** (etiqueta `without_mask`).
    *   Personas con **mascarilla mal puesta** (etiqueta `mask_worn_incorrectly`).
*   Visualización de las detecciones con cuadros delimitadores y etiquetas de clase con códigos de color en el feed de la cámara.
*   Utiliza un modelo EfficientDet-D1 afinado para esta tarea específica, cargado localmente.

## Requisitos Previos

*   Python 3.8 o superior (probado con Python 3.12).
*   Una cámara web conectada y funcionando.
*   Git (para clonar el repositorio del modelo).

## Configuración del Proyecto

1.  **Clona este repositorio (si aún no lo has hecho):**
    Si estás obteniendo este proyecto desde GitHub, clónalo:
    ```bash
    git clone <URL_DE_ESTE_REPOSITORIO>
    cd <NOMBRE_DE_LA_CARPETA_DEL_PROYECTO>
    ```

2.  **Descarga el modelo pre-entrenado (¡Paso Obligatorio!):**
    Este proyecto **no incluye** los archivos del modelo de detección de mascarillas debido a su tamaño. Debes clonar el repositorio `coolmunzi/face_mask_detector` **dentro del directorio raíz de este proyecto**.
    
    Desde el directorio raíz de este proyecto, ejecuta:
    ```bash
    git clone https://github.com/coolmunzi/face_mask_detector.git
    ```
    Esto creará una carpeta llamada `face_mask_detector`.

3.  **Verifica la estructura de carpetas:**
    Después del paso anterior, la estructura de carpetas relevante para el modelo debe ser:
    ```
    TU_PROYECTO_RAIZ/
    ├── face_mask_detector/  <-- Clon del repo de coolmunzi
    │   └── inference-graph/
    │       └── saved_model/
    │           ├── variables/
    │           └── saved_model.pb
    ├── main.py
    ├── requirements.txt
    ├── LICENSE
    └── README.md
    ```
    **Importante:** La variable `PATH_TO_SAVED_MODEL` en `main.py` está configurada como `"face_mask_detector/inference-graph/saved_model"`. Si la estructura es diferente, deberás ajustar esta ruta.

4.  **Crea un entorno virtual (recomendado):**
    ```bash
    python -m venv venv
    ```
    Actívalo:
    *   En Windows:
        ```bash
        .\venv\Scripts\activate
        ```
    *   En macOS/Linux:
        ```bash
        source venv/bin/activate
        ```

5.  **Instala las dependencias del proyecto:**
    Asegúrate de que tu entorno virtual esté activado. Luego, instala las dependencias para este script detector:
    ```bash
    pip install -r requirements.txt
    ```
    El archivo `requirements.txt` de este proyecto debe contener al menos:
    ```
    tensorflow>=2.12.0
    opencv-python
    ```
    No es necesario instalar las dependencias del `requirements.txt` del repositorio `coolmunzi` para ejecutar `main.py`, ya que solo usamos su `SavedModel`.

## Uso

Una vez que el modelo esté en la ubicación correcta y las dependencias estén instaladas, ejecuta el script principal desde el directorio raíz del proyecto:

```bash
python main.py
```

Se abrirá una ventana mostrando el feed de tu cámara web con las detecciones de mascarillas.

*   Presiona la tecla **'q'** para cerrar la aplicación.

## Clases Detectadas y Colores

El modelo identifica las siguientes clases, mostradas con los siguientes colores:

*   `with_mask`: Persona usando mascarilla correctamente (Verde).
*   `without_mask`: Persona sin mascarilla (Rojo).
*   `mask_worn_incorrectly`: Persona con la mascarilla mal puesta (Naranja).

## Notas Adicionales

*   El rendimiento de la detección puede variar según las condiciones de iluminación, la calidad de la cámara y la distancia a la cámara.
*   El umbral de confianza para las detecciones (`DEFAULT_SCORE_THRESHOLD`) está configurado en `0.5` en `main.py`. Puedes ajustarlo si es necesario.
*   El tamaño de entrada de imagen esperado por el modelo EfficientDet-D1 actual es `640x640` (`INPUT_SIZE` en `main.py`).
*   Este proyecto se basa en el trabajo y el modelo pre-entrenado de [coolmunzi/face_mask_detector](https://github.com/coolmunzi/face_mask_detector).

## Solución de Problemas Comunes

*   **"Error: No se pudo acceder a la cámara."**: Verifica que tu cámara esté conectada, no esté siendo utilizada por otra aplicación, y que el índice de cámara en `cv2.VideoCapture(0, cv2.CAP_DSHOW)` (el `0`) sea el correcto. `CAP_DSHOW` se usa para mejorar la compatibilidad en Windows.
*   **"Error al cargar el modelo..." o "PATH_TO_SAVED_MODEL no ha sido configurada"**: Asegúrate de haber clonado el repositorio `coolmunzi/face_mask_detector` dentro de la carpeta de este proyecto y que la ruta `PATH_TO_SAVED_MODEL` en `main.py` (`"face_mask_detector/inference-graph/saved_model"`) sea correcta.
*   **Error `was expected to be a uint8 tensor but is a float tensor` (o viceversa)**: La función `preprocess_frame` en `main.py` está configurada para el tipo de tensor de entrada que espera el `SavedModel` actual (`uint8`). Si cambias de modelo, podrías necesitar ajustar el tipo de datos (`tf.uint8` o `tf.float32`) y la normalización en esta función.
*   **Clases detectadas incorrectamente (ej. "con máscara" cuando no la hay)**: El diccionario `CLASS_NAMES` en `main.py` mapea los IDs de clase del modelo a nombres legibles. Si las detecciones son consistentemente erróneas (pero los cuadros aparecen), podría ser necesario ajustar este mapeo según cómo el modelo específico fue entrenado y exportado. (Este problema ya debería estar corregido según las últimas modificaciones).

## Contribuciones

Las contribuciones son bienvenidas. Por favor, abre un *issue* para discutir cambios importantes o envía un *pull request*.

## Licencia

Este proyecto se distribuye bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles. 
