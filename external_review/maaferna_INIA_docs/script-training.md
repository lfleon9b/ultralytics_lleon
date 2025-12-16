Documentación del Script de Entrenamiento para YOLOv8
Resumen

Este script en Python estandariza y automatiza el proceso de entrenamiento para modelos YOLOv8, proporcionando un enfoque estructurado para experimentar con diferentes configuraciones, gestionar resultados y rastrear el desempeño utilizando ClearML para la gestión de experimentos. Está diseñado para soportar configuraciones multi-GPU y garantizar la reproducibilidad mediante el uso de semillas aleatorias consistentes y el registro de todos los parámetros y métricas relevantes.

Principales Funcionalidades

1. Configuración del Experimento e Inicialización

El script permite configurar y ejecutar múltiples entrenamientos con varias combinaciones de hiperparámetros, incluyendo:

    Versión del modelo: Seleccionar entre las versiones de YOLOv8 (n, s, m, l, x).
    Tamaño de imagen: Especificar el tamaño de entrada para el entrenamiento (e.g., 416, 640, 1024).
    Tamaño de batch: Definir el número de muestras por lote según la capacidad de memoria de la GPU.
    Dispositivos: Soporte para configuraciones multi-GPU (e.g., ['0', '1']).
    Número de ejecuciones: Realizar múltiples corridas independientes para validación estadística.
    Semilla aleatoria: Garantizar reproducibilidad para cada ejecución mediante la inicialización de una semilla consistente.

El script utiliza el marco ClearML para rastrear las tareas y resultados de cada entrenamiento. Se crea una tarea de ClearML para cada ejecución, lo que permite registrar logs detallados y artefactos.

2. Ejecución del Entrenamiento

Cada ejecución de entrenamiento realiza los siguientes pasos:

    Creación de la Tarea en ClearML: Se inicializa una tarea única en ClearML para cada ejecución con metadatos como versión del modelo, tamaño de imagen y el identificador de la ejecución.

    Carga del Modelo: Los pesos del modelo YOLOv8 se cargan desde un directorio predefinido (models/).

    Configuración del Entrenamiento: Los hiperparámetros clave como epochs, batch size y image size se configuran dinámicamente para cada corrida.

    Entrenamiento Multi-GPU: El script utiliza CUDA_VISIBLE_DEVICES para configuraciones multi-GPU, sin implementar Distributed Data Parallel (DDP) por incompatiblidad con Windows OS. Actualmente las GPU trabajan de manera independientes desde dos terminales diferentes.

    Rastreo de Resultados: YOLOv8 genera métricas clave como mAP@50, Precisión y Recall. Estas métricas se registran en ClearML y en archivos JSON para cada ejecución.

    Manejo de Semilla Aleatoria: Se asigna una semilla única basada en el tiempo para garantizar la diversidad entre ejecuciones.

3. Registro y Gestión de Resultados

Para cada ejecución:

    Se registra un resumen JSON con información clave como:

        Hiperparámetros (e.g., tamaño de batch, tamaño de imagen, epochs, dispositivo).
        Resultados obtenidos (mAP@50, F1-score, Precision, Recall).
        Semilla utilizada.
        Archivos generados como pesos del modelo (best.pt), curvas de precisión-recall y matrices de confusión.

    Todos los resultados se consolidan en un archivo JSON al final de todas las corridas para facilitar el análisis comparativo.

4. Lógica para la Estandarización

    Control de Semillas: Cada corrida utiliza una semilla aleatoria única para reproducibilidad, la cual queda registrada en el JSON para reproductibilidad del experimento.
    Resultados Detallados: Las métricas de cada corrida se documentan y suben a ClearML.
    Artefactos y Configuración: Se suben los pesos del modelo, configuraciones y resultados como artefactos en ClearML.

Ventajas del Script

    Automatización: Reduce la intervención manual al ejecutar múltiples experimentos.
    Reproducibilidad: Controla y registra semillas aleatorias para asegurar consistencia.
    Flexibilidad: Soporte para múltiples GPUs y configuraciones dinámicas.
    Trazabilidad: Integra ClearML para facilitar la comparación entre ejecuciones.

Uso Sugerido

El script es adecuado para proyectos de investigación o producción donde se necesita explorar diferentes configuraciones de modelos, tamaños de imágenes y otros parámetros, mientras se asegura una trazabilidad robusta y reproducibilidad de resultados.