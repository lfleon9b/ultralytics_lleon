Justificación de la Distribución de Datos en el Proyecto

En el presente proyecto, se define la proporción de datos para entrenamiento, validación y prueba basándose en las imágenes originales del conjunto de datos, utilizando la metodología de distribución 80%-10%-10% ampliamente aceptada en la literatura científica. Este enfoque asegura la validez y la reproducibilidad de los resultados obtenidos en las etapas de entrenamiento y evaluación del modelo.

Distribución Basada en Imágenes Originales

El conjunto de datos está compuesto por un total de 444 imágenes originales. Estas imágenes se distribuyen de la siguiente manera:

    Entrenamiento (80%): 356 imágenes originales, utilizadas para construir el modelo.
    Validación (10%): 44 imágenes originales, utilizadas para medir el desempeño del modelo durante el entrenamiento y realizar ajustes de hiperparámetros.
    Prueba (10%): 44 imágenes originales, empleadas exclusivamente para evaluar el modelo final en datos no vistos.

Para enriquecer el conjunto de datos y aumentar la variabilidad durante el entrenamiento, se emplea aumento de datos exclusivamente sobre las imágenes de entrenamiento, generando un total de 1780 imágenes aumentadas. Esto permite que el modelo aprenda patrones más robustos sin afectar los conjuntos de validación y prueba.

Importancia de la Distribución Basada en Imágenes Originales

La decisión de no aplicar aumento de datos en los conjuntos de validación y prueba se sustenta en varias razones científicas:

    Evitar sesgos durante la evaluación: Al mantener las imágenes originales en validación y prueba, se asegura que estos conjuntos reflejen un desempeño realista del modelo en datos no vistos, sin influencias de los datos aumentados que podrían sesgar los resultados.

    Reproducibilidad: Los conjuntos de validación y prueba permanecen constantes y no dependen de técnicas de aumento que podrían variar entre experimentos.

    Buena práctica en la literatura: La mayoría de los estudios y guías en aprendizaje profundo recomiendan mantener validación y prueba sin aumentos, ya que estas etapas están destinadas a evaluar el modelo en datos representativos y sin alteraciones [1, 2].

Metodología y Referencias

El enfoque descrito sigue las recomendaciones generales en la literatura científica sobre aprendizaje profundo y visión por computadora:

    Goodfellow et al. (2016) destacan que los conjuntos de validación y prueba deben reflejar fielmente el comportamiento del modelo en el mundo real y no deben contener datos artificialmente aumentados, ya que esto podría introducir sesgos [1].

    Howard y Gugger (2020) sugieren que el aumento de datos es una herramienta poderosa para entrenamiento, pero deben excluirse de validación y prueba para mantener la integridad de los resultados [2].
    
    Según Ronneberger et al. (2015), en proyectos con cantidades limitadas de datos originales, como en tu caso, es crucial mantener proporciones claras y evitar alteraciones en los conjuntos de validación y prueba para asegurar evaluaciones confiables [3].

    Ventajas del Enfoque

    Validez científica: Garantiza que las métricas de validación y prueba reflejen el desempeño real del modelo en datos no vistos.
    Generalización: La evaluación no está influenciada por datos aumentados que podrían estar sobrerrepresentados.
    Consistencia y reproducibilidad: Facilita la comparación entre experimentos y la replicación de resultados en otros estudios.

Conclusión

El enfoque utilizado en este proyecto es científicamente válido y común en investigaciones en visión por computadora. La distribución 80%-10%-10% basada en las imágenes originales y la aplicación de aumento de datos únicamente en el conjunto de entrenamiento asegura un balance óptimo entre la robustez del modelo y la validez de las métricas de evaluación.

Referencias

    Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.
    Howard, J., & Gugger, S. (2020). Deep Learning for Coders with fastai and PyTorch. O'Reilly Media.
    Ronneberger, O., Fischer, P., & Brox, T. (2015). "U-Net: Convolutional Networks for Biomedical Image Segmentation." Medical Image Computing and Computer-Assisted Intervention (MICCAI).