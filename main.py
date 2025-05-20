import tensorflow as tf
# import tensorflow_hub as hub # Ya no se usa TF Hub para este modelo
import numpy as np
import cv2
from typing import Dict, Tuple, Any

# 1. Constantes del Modelo y Aplicación
# !!! IMPORTANTE: Ajusta esta ruta al directorio 'saved_model' del modelo clonado !!!
# Ejemplo: 'coolmunzi_face_mask_detector/inference-graph/saved_model'
PATH_TO_SAVED_MODEL: str = "face_mask_detector/inference-graph/saved_model" 

# Clases del modelo coolmunzi/face_mask_detector
# Asumiendo IDs 1, 2, 3 según el README y la práctica común de TFODAPI
# Se invierten las clases 1 y 2 según la observación del usuario
CLASS_NAMES: Dict[int, str] = {
    1: "without_mask",  # El modelo parece devolver ID 1 para 'sin máscara'
    2: "with_mask",     # El modelo parece devolver ID 2 para 'con máscara'
    3: "mask_worn_incorrectly"
}
# INPUT_SIZE podría necesitar ajuste para EfficientDet-D1. 
# Consulta el pipeline.config del modelo clonado. Por ahora, usamos 512.
INPUT_SIZE: int = 640 
DEFAULT_SCORE_THRESHOLD: float = 0.5 # Umbral de confianza

# Colores para la visualización (formato BGR para OpenCV)
COLOR_WITH_MASK: Tuple[int, int, int] = (0, 255, 0)    # Verde
COLOR_WITHOUT_MASK: Tuple[int, int, int] = (0, 0, 255) # Rojo
COLOR_MASK_WORN_INCORRECTLY: Tuple[int, int, int] = (0, 165, 255) # Naranja
COLOR_UNKNOWN: Tuple[int, int, int] = (128, 128, 128) # Gris

# 2. Cargar modelo local (SavedModel)
def load_local_model(model_path: str) -> Any:
    """Carga un SavedModel de TensorFlow desde una ruta local."""
    print(f"Cargando modelo desde: {model_path}")
    # Eliminamos la advertencia específica de la ruta no actualizada, ya que la estamos estableciendo.
    # if model_path == "ruta/a/tu/coolmunzi_face_mask_detector/inference-graph/saved_model":
    #     print("ADVERTENCIA: La ruta del modelo no ha sido actualizada. Por favor, edita PATH_TO_SAVED_MODEL en main.py.")
    #     raise ValueError("PATH_TO_SAVED_MODEL no ha sido configurada. Por favor, actualiza la ruta en main.py.")

    try:
        loaded_model = tf.saved_model.load(model_path)
    except OSError as e:
        print(f"Error al cargar el modelo desde '{model_path}': {e}")
        print("Asegúrate de que la ruta es correcta y el modelo existe en ese directorio.")
        raise
    
    # Los modelos de TF Object Detection API suelen tener una signatura 'serving_default'
    inference_fn = loaded_model.signatures['serving_default']
    print("Modelo cargado exitosamente.")
    return inference_fn

detector_inference_fn = load_local_model(PATH_TO_SAVED_MODEL)

# 3. Función de preprocesado
def preprocess_frame(frame: np.ndarray, input_size: int = INPUT_SIZE) -> tf.Tensor:
    """
    Redimensiona el frame al tamaño de entrada del modelo y asegura el tipo uint8.
    Devuelve un tf.Tensor.
    """
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (input_size, input_size))
    # Añadir dimensión de batch
    input_tensor_np = np.expand_dims(img_resized, axis=0)
    # Convertir a tf.Tensor uint8 como lo espera este SavedModel específico
    input_tensor = tf.convert_to_tensor(input_tensor_np, dtype=tf.uint8)
    return input_tensor

# 4. Bucle de inferencia y visualización en tiempo real
def realtime_mask_detection(score_thresh: float = DEFAULT_SCORE_THRESHOLD) -> None:
    """
    Captura video de la webcam, realiza inferencia y muestra los resultados.
    """
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print("Error: No se pudo acceder a la cámara. Verifique la conexión o el índice de la cámara.")
        return

    print("Cámara iniciada. Presione 'q' para salir.")
    print(f"Usando umbral de confianza: {score_thresh}")
    print(f"Clases esperadas: {CLASS_NAMES}")


    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo leer el frame de la cámara.")
            break

        original_height, original_width = frame.shape[:2]
        input_tensor = preprocess_frame(frame, INPUT_SIZE)

        # Inferencia usando la función cargada del SavedModel
        results = detector_inference_fn(input_tensor)
        
        # El formato de salida de los modelos de TF Object Detection API es ligeramente diferente
        # y las clases suelen estar como float, necesitan conversión a int.
        # El número de detecciones válidas también es importante.
        num_detections = int(results.pop('num_detections'))
        detection_boxes = results['detection_boxes'][0,:num_detections].numpy()
        detection_scores = results['detection_scores'][0,:num_detections].numpy()
        # Las clases pueden ser float, convertir a int. A veces están indexadas +1.
        # Para los modelos de TFODAPI, las clases suelen estar ya en el rango correcto (ej. 1-N).
        detection_classes = results['detection_classes'][0,:num_detections].numpy().astype(np.int64)

        # print(f"--- Frame --- Detecciones crudas (top 5): Scores: {detection_scores[:5]}, Clases: {detection_classes[:5]}") # DEBUG

        found_detection_above_threshold = False
        for i in range(num_detections): # Iterar solo sobre las detecciones válidas
            score = detection_scores[i]
            if score < score_thresh:
                # Las detecciones suelen estar ordenadas por score, así que podemos romper antes.
                break 
            
            found_detection_above_threshold = True
            box = detection_boxes[i]
            class_id = detection_classes[i] 

            # Convertir coordenadas normalizadas (y_min, x_min, y_max, x_max) a píxeles
            y_min, x_min, y_max, x_max = box
            x1 = int(x_min * original_width)
            y1 = int(y_min * original_height)
            x2 = int(x_max * original_width)
            y2 = int(y_max * original_height)

            class_name = CLASS_NAMES.get(class_id, f"ClaseID:{class_id}")
            
            # print(f"  Detectado: {class_name} (ID:{class_id}) con score: {score:.2f}") # DEBUG

            box_color = COLOR_UNKNOWN
            if class_name == "with_mask":
                box_color = COLOR_WITH_MASK
            elif class_name == "without_mask":
                box_color = COLOR_WITHOUT_MASK
            elif class_name == "mask_worn_incorrectly":
                box_color = COLOR_MASK_WORN_INCORRECTLY
            
            label = f"{class_name}: {score:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            y1_label = max(y1, label_size[1] + 10)
            cv2.rectangle(frame, (x1, y1_label - label_size[1] - 10), 
                          (x1 + label_size[0], y1_label - base_line - 10), box_color, cv2.FILLED)
            cv2.putText(frame, label, (x1, y1_label - 7), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # if not found_detection_above_threshold and num_detections > 0 and detection_scores[0] > 0.01:
            # print(f"INFO: Max score ({detection_scores[0]:.2f} para ClaseID:{detection_classes[0]}) no supera umbral de {score_thresh:.2f}")
            
        cv2.imshow("Detector de Mascarillas - EfficientDet-D1", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Saliendo...")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Recursos liberados.")

if __name__ == "__main__":
    print("Iniciando detector de mascarillas...")
    print("Asegúrate de haber clonado 'coolmunzi/face_mask_detector' y de que PATH_TO_SAVED_MODEL esté bien configurada.")
    try:
        realtime_mask_detection()
    except ValueError as ve: # Captura el error específico si la ruta no está configurada
        print(f"Error de configuración: {ve}")
    except Exception as e:
        print(f"Ocurrió un error durante la ejecución: {e}")
        # Considera un logging más robusto
