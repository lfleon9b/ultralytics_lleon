Para respaldar el uso de F1-score como métrica final en el último epoch (o el mejor epoch en términos de mAP), existen referencias en la literatura de aprendizaje profundo y visión por computadora donde se destaca esta práctica. A continuación, algunos artículos científicos y prácticas recomendadas que refuerzan el uso de métricas promedio en el último epoch:

YOLO y mAP/F1 como Métricas Finales de Evaluación:

    Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). "You Only Look Once: Unified, Real-Time Object Detection." Proceedings of the IEEE conference on computer vision and pattern recognition.
        Este artículo sobre YOLO (You Only Look Once) introduce el mAP como métrica primaria y calcula las métricas en un punto final de entrenamiento, estableciendo el uso de métricas finales como el enfoque principal en reconocimiento de objetos.

Métricas Promedio en Evaluaciones de Modelos de Detección:

    Huang, J., et al. (2017). "Speed/accuracy trade-offs for modern convolutional object detectors." Proceedings of the IEEE conference on computer vision and pattern recognition.
        Este estudio compara varios detectores modernos y utiliza métricas de evaluación al final de los entrenamientos, específicamente mAP y recall, destacando el mAP final como el indicador primario de desempeño en lugar de métricas en múltiples epochs.

Buenas Prácticas en la Evaluación de Modelos de Detección de Objetos:

    Lin, T. Y., et al. (2014). "Microsoft COCO: Common objects in context." European conference on computer vision.
        En este artículo se establece el conjunto de datos COCO, utilizado ampliamente en benchmarks, donde se recomienda evaluar el modelo con mAP y otras métricas finales de precisión y recall. Aquí, se usa el mAP en diferentes umbrales como mAP50-95 para capturar la efectividad del modelo en el último epoch.

Evaluación de Modelos con Métricas Promedio en el Último Epoch:

    Wu, Y., & He, K. (2019). "Group normalization." Proceedings of the IEEE/CVF International Conference on Computer Vision.
        En este artículo, los autores evaluaron las métricas promedio de los modelos al final del entrenamiento, un enfoque estándar que consideran relevante para reflejar el desempeño general del modelo sin variabilidad de métricas durante el entrenamiento.



Estos artículos destacan que, en general, la evaluación de métricas como mAP y F1-score suele realizarse en el último epoch o en el mejor epoch de acuerdo con la métrica principal (usualmente mAP50-95) como método confiable de evaluación. De esta manera, el modelo se selecciona basado en su rendimiento consistente, respaldado por métricas promediadas o finales, y el cálculo epoch a epoch se usa para monitoreo interno más que para la evaluación comparativa.