Procedimiento de Selección del Mejor Modelo Basado en mAP50-95 y F1-Score

Proyecto: Reconocimiento de Imágenes mediante Deep Learning


1. Introducción y Objetivo

En el contexto de proyectos de detección de objetos, como es el caso de este estudio de reconocimiento de imágenes, es crucial evaluar los modelos utilizando métricas que capturen tanto la precisión en la detección como la localización precisa de los objetos en las imágenes. Este documento describe el procedimiento de selección del mejor modelo, basado en una combinación ponderada de dos métricas clave: mAP50-95 y F1-score.

La elección de un promedio ponderado entre estas métricas permite obtener un balance adecuado entre la precisión de detección de objetos y la adaptabilidad del modelo a múltiples umbrales de IoU (Intersection over Union), mejorando así su generalización en el entorno de producción.

2. Decisión del Procedimiento

Métricas Seleccionadas

Las métricas de evaluación seleccionadas para determinar el mejor modelo en este proyecto son:

    mAP50-95: Esta métrica evalúa la precisión promedio del modelo en un rango de umbrales de IoU, desde 50% hasta 95% en incrementos del 5%. Esto proporciona una visión integral del desempeño del modelo en términos de precisión de localización.

    F1-score: El F1-score proporciona un equilibrio entre precisión (proporción de verdaderos positivos en las detecciones) y recall (capacidad del modelo para detectar todos los objetos presentes). Esto es especialmente importante en aplicaciones donde se necesita un balance entre minimizar los falsos negativos y los falsos positivos.

Justificación del Promedio Ponderado

El promedio ponderado se elige para priorizar las métricas en función de su relevancia en el contexto de este proyecto:

    mAP50-95 tiene un peso significativo en aplicaciones donde la precisión en la localización de los objetos es crítica. Por lo tanto, en este proyecto, se asigna un peso ligeramente mayor a mAP50-95.

    F1-score sigue siendo esencial, pues asegura que el modelo mantenga un buen equilibrio entre precisión y recall, reduciendo tanto los falsos positivos como los falsos negativos.

Cálculo del Score Ponderado

Para cada ejecución del modelo, el score ponderado será calculado de la siguiente manera:

Score ponderado=(α⋅mAP50-95)+(β⋅F1-score)
Score ponderado=(α⋅mAP50-95)+(β⋅F1-score)

Donde:

    αα y ββ son los pesos asignados a mAP50-95 y F1-score, respectivamente.
    La suma de los pesos debe ser igual a 1 (α+β=1α+β=1).

Para este proyecto, se proponen los siguientes valores:

    α=0.7α=0.7: Mayor peso al mAP50-95 para priorizar la precisión de localización.
    β=0.3β=0.3: Peso del F1-score para garantizar un balance adecuado entre precisión y recall.

3. Registro de Indicadores por Ejecución

Para evaluar de manera efectiva cada configuración de modelo, es fundamental registrar todas las métricas relevantes en cada ejecución. Los siguientes indicadores deben registrarse para cada combinación de parámetros del modelo:

    mAP50: Precisión media a un IoU de 50%.
    mAP50-95: Precisión media en un rango de IoU de 50% a 95%.
    F1-score: Promedio del equilibrio entre precisión y recall en el umbral de confianza seleccionado.
    Recall (Recall Valid Mean): Capacidad del modelo para detectar todos los objetos presentes en las imágenes de validación.
    Precisión (Precision Valid Mean): Proporción de verdaderos positivos entre todas las detecciones realizadas en validación.

Estas métricas permitirán realizar un análisis detallado de la performance del modelo, y a partir de ellas se calculará el score ponderado para seleccionar el mejor modelo.

4. Procedimiento de Selección del Mejor Modelo

    Entrenamiento y Registro de Métricas:
        Para cada configuración de tamaño de imagen, modelo, y threshold de confianza, se entrenará el modelo y se registrarán las métricas mencionadas anteriormente.
    Cálculo del Score Ponderado:
        Para cada ejecución, se calculará el score ponderado utilizando la fórmula descrita:
        Score ponderado=(0.7⋅mAP50-95)+(0.3⋅F1-score)
        Score ponderado=(0.7⋅mAP50-95)+(0.3⋅F1-score)
    Selección del Mejor Modelo:
        El modelo con el score ponderado más alto será considerado el mejor para esa configuración específica de parámetros.
        Para aumentar la robustez, se evaluará el mejor modelo en el conjunto de test para confirmar que los resultados se mantengan en datos no vistos.

5. Justificación Científica

La combinación de mAP50-95 y F1-score es una práctica recomendada en la literatura en visión computacional y detección de objetos, al equilibrar precisión y generalización. A continuación se citan estudios relevantes:

    Zhao et al., "Object Detection with Deep Learning: A Review" (2019). IEEE Transactions on Neural Networks and Learning Systems.
    Este artículo revisa técnicas de detección de objetos y destaca la importancia de utilizar múltiples métricas, incluyendo mAP y F1-score, para una evaluación integral.

    Lin et al., "Feature Pyramid Networks for Object Detection" (2017). Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).
    Este estudio utiliza mAP como métrica de referencia para evaluar la localización precisa y destaca la importancia de evaluar el recall para una detección efectiva de objetos.

    Huang et al., "Speed/Accuracy Trade-offs for Modern Convolutional Object Detectors" (2017). CVPR.
    Discute el balance entre precisión y recall para modelos en aplicaciones en tiempo real, proponiendo el uso de mAP y F1-score como métricas balanceadas.

    Redmon & Farhadi, "YOLOv3: An Incremental Improvement" (2018). arXiv preprint arXiv:1804.02767.
    En este artículo, los autores destacan el uso de mAP para evaluar la precisión del modelo en diferentes tamaños de IoU, indicando la relevancia de esta métrica para aplicaciones que requieren alta precisión en localización.

    Li et al., "Generalizing YOLO for Object Detection on UAV Images" (2020). Remote Sensing.
    Este estudio destaca la importancia de ajustar las métricas de evaluación en función de la aplicación específica y justifica el uso de mAP junto con recall y F1-score.

6. Conclusión

El enfoque de promedio ponderado entre mAP50-95 y F1-score es consistente con prácticas avanzadas en la literatura para evaluación en proyectos de detección de objetos. Este procedimiento asegura que el modelo final cumpla con los estándares de precisión y eficiencia necesarios para su aplicación en producción, optimizando tanto la localización como la capacidad de detección en los objetos del dataset.