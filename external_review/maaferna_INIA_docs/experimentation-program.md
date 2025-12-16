Programa de Experimentación para el Reconocimiento de Imágenes con YOLOv8

Proyecto: Reconocimiento de Imágenes con Deep Learning Maleza INIA FIA

Fecha de Inicio: 

1. Objetivo del Experimento

El propósito de este programa de experimentación es entrenar y optimizar el modelo YOLOv8 para el reconocimiento de imágenes con alta densidad de objetos, con diferentes resoluciones de imagenes. Con el objetivo de obtener un balance en las métricas de precisión y recall, así como evaluar el impacto de diferentes configuraciones de parámetros sobre la calidad de los resultados de inferencia. Este experimento se centrará en explorar las mejores configuraciones considerando el tamaño de imagen, confident threshold y el número máximo de detecciones.

2. Configuración del Experimento

2.1 Estructura de Datos

Distribución de Datos:

    Entrenamiento: 80% de las imágenes
    Validación: 10% de las imágenes
    Prueba: 10% de las imágenes

Total de Imágenes: 1780 (entrenamiento), 223 (validación y prueba combinados)

2.2 Parámetros Experimentales

    Modelos (Backbones) de YOLOv8:
        yolov8n (nano)
        yolov8s (small)
        yolov8m (medium)
        yolov8l (large)
        yolov8x (extra large)

    Tamaños de Imagen:
        416x416
        640x640
        1024x1024
        2048x2048

    Max Detections (max_det):

        500 detecciones por imagen

    Confidence Threshold:
        0.3 para maximizar el recall en escenarios de una alta densidad de objectos para identificar.
        0.5 para mejorar la precisión en imágenes con objetos bien definidos, con un modelo más conservador.

2.3 Data Augmentation

    Mosaic y Horizontal Flip habilitados para mejorar la variabilidad del dataset y la generalización del modelo sin afectar significativamente los tiempos de entrenamiento. Estas funcionalidades son incluidas por defecto en los modelos YOLO, por lo que, no es necesario ajustes adicionales en el código.

3. Metodología Experimental

3.1 Variables a Evaluar

Cada combinación de modelo, tamaño de imagen y confident threshold será evaluada en función de los siguientes indicadores:

    mAP50: Precisión media a un IoU de 50%
    mAP50-95: Precisión media a múltiples umbrales de IoU (Intersection over Union), en tramos del 5%
    Recall: Capacidad del modelo para encontrar todos los objetos en la imagen
    Precisión: Proporción de verdaderos positivos en las detecciones realizadas
    F1-score: Métrica que combina precisión y recall en una sola puntuación, proporcionando un equilibrio entre ambas para evaluar la exactitud y exhaustividad del modelo.

3.2 Combinaciones de Configuración

Dado que existen 5 modelos, 4 tamaños de imagen, 2 valores de threshold y una configuración única para max_det, se generarán las siguientes combinaciones:

    5 modelos × 4 tamaños de imagen × 2 thresholds = 40 combinaciones

Cada combinación será ejecutada en 5 corridas independientes para asegurar la estabilidad de los resultados y reducir la variabilidad en las métricas finales.

Total de Corridas: 200 experimentos.

3.3 Procedimiento Experimental

    Preparación del Dataset:
        Dividir el dataset según la proporción 80/10/10 para entrenamiento, validación y prueba.
        Asegurarse de que el dataset esté correctamente balanceado y limpio de errores en las etiquetas.

    Entrenamiento de los Modelos:
        Ejecutar cada combinación de modelo, tamaño de imagen y threshold en el entorno de entrenamiento.
        Registrar las métricas mAP, recall y precisión para cada combinación.

    Análisis de Resultados:
        Calcular el promedio y la desviación estándar de las métricas de precisión y recall para cada configuración.
        Comparar los resultados obtenidos en cada configuración y seleccionar las mejores combinaciones en función del mAP50 y mAP50-95.

    Documentación y Reporte:
        Detallar los resultados de cada experimento en un archivo de seguimiento.
        Generar un reporte con gráficos y tablas que ilustren el desempeño de cada configuración de modelo y sus métricas de rendimiento.

4. Cronograma

    Preparación del Dataset: 1 semana
    Ejecución de Experimentos: 4 semanas
    Análisis de Resultados: 1 semana
    Elaboración de Reporte Final: 1 semana

Duración Total Estimada: 7 semanas

5. Referencias y Fundamentación Científica

Referencias Bibliográficas:

    Bochkovskiy, A., Wang, C. Y., & Liao, H. Y. M. (2020). YOLOv4: Optimal speed and accuracy of object detection. arXiv preprint arXiv:2004.10934.
    Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., et al. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. International Conference on Learning Representations (ICLR).

Estas referencias respaldan la importancia de ajustar los parámetros de entrenamiento y el uso de un conjunto de validación adecuado para mejorar la precisión y evitar el sobreajuste.