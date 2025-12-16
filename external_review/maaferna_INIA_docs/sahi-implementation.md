La implementación de SAHI (Slicing Aided Hyper Inference) ha demostrado mejoras significativas en la detección de objetos pequeños y en imágenes de alta resolución. Según el estudio "Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection" de Akyon et al. (2022), la aplicación de SAHI incrementó la precisión promedio (AP) en un 6.8% para el detector FCOS, 5.1% para VFNet y 5.3% para TOOD, sin necesidad de ajustes adicionales en los modelos.
ArXiv

Además, la investigación "Evaluation of YOLO Models with Sliced Inference for Small Object Detection" de Keles et al. (2022) evaluó modelos YOLOv5 y YOLOX en el dataset VisDrone2019Det. Los resultados mostraron que la inferencia con imágenes segmentadas aumentó el AP50 en todos los experimentos, siendo más notable en los modelos YOLOv5. La combinación de fine-tuning e inferencia segmentada produjo mejoras sustanciales, alcanzando un AP50 de 48.8 en el modelo YOLOv5-Large.
ArXiv

Estos estudios respaldan la eficacia de SAHI para mejorar la precisión en la detección de objetos pequeños y en imágenes de alta resolución, destacando su aplicabilidad en proyectos que requieren análisis detallados en entornos agrícolas.

Anexo 4: Metodología y Procedimiento Técnico para el Uso de SAHI
Este anexo describe la metodología y lógica empleadas para implementar SAHI (Slicing Aided Hyper Inference) en el análisis de imágenes de alta resolución, optimizando la precisión de las inferencias y su utilidad práctica en escenarios agrícolas.

4.1. Selección del Modo de Ejecución

El flujo de trabajo permite seleccionar entre dos opciones de procesamiento:
Procesar una sola imagen: Ideal para imágenes individuales con alto nivel de detalle.
Procesar un directorio completo: Diseñado para flujos en lotes, donde se procesan múltiples imágenes en una sola ejecución.
Se definen parámetros clave que guían el procesamiento:
Versión del modelo: Selección de la versión del modelo, (e.g., YOLOv8) y el tamaño (e.g., n, s, m, l, x) más adecuado según las métricas de entrenamiento.
Tamaño de imagen: Corresponde al tamaño de entrada definido durante el entrenamiento del modelo, garantizando que las condiciones de inferencia sean compatibles con las características aprendidas por el modelo. Aunque existe la opción de modificar este tamaño para realizar pruebas o adaptarlo a necesidades específicas, no se recomienda hacerlo, ya que puede afectar negativamente la precisión de las predicciones.
Directorio predeterminado: La estructura de los datos generados (outputs de entrenamientos y modelos) sigue una lógica estandarizada, donde los modelos y resultados se organizan en carpetas definidas según configuraciones clave como tamaño de imagen, versión del modelo y otros parámetros. Esto garantiza un acceso eficiente y facilita el análisis de los datos al mantener un orden lógico y consistente en el proyecto.

4.2. Identificación del Mejor Modelo

La metodología identifica el modelo best.pt óptimo para la inferencia, basado en métricas clave:
    mAP50-95: Precisión promedio en detección, evaluando la consistencia del modelo.
    F1 score: Equilibrio entre precisión y recall, asegurando detecciones fiables.

El modelo seleccionado garantiza que las inferencias sean realizadas bajo las mejores condiciones posibles.

4.3 Configuración de Parámetros de SAHI

Para asegurar una inferencia eficiente, se configuran parámetros específicos:
Tamaño de los cortes (slice_size): Cada fragmento procesado corresponde al tamaño de entrada del modelo utilizado durante el entrenamiento.
Superposición entre cortes (overlap_ratio): Minimiza errores en las predicciones cerca de los bordes de los fragmentos.
Umbral de confianza: Filtra predicciones con baja probabilidad, mejorando la precisión general.
Impacto del Tamaño de los Cortes (slice_size) en el Rendimiento
El tamaño de los cortes (slice_size) afecta tanto el rendimiento del modelo como la calidad de las predicciones. Sin embargo, los resultados dependen directamente del tamaño de entrada en el que el modelo fue entrenado y de cómo se configuran los cortes durante la inferencia. A continuación, se aclaran algunos conceptos importantes:
Cortes más grandes que el tamaño de entrada del modelo:
Cuando los cortes son más grandes que el tamaño de entrada, el modelo puede perder precisión en las predicciones porque no está adaptado para procesar esas dimensiones completas.
Esto puede acelerar el procesamiento al reducir el número de cortes necesarios, pero con un riesgo significativo de pérdida de precisión.
Cortes más pequeños que el tamaño de entrada del modelo:
Utilizar cortes más pequeños no garantiza mejores resultados, especialmente si el modelo fue entrenado con un tamaño mayor. Esto ocurre porque el modelo está optimizado para procesar características en un contexto más amplio, y reducir el contexto puede afectar negativamente la capacidad del modelo para identificar objetos correctamente.
Además, la generación de cortes más pequeños incrementa el tiempo de procesamiento y no siempre justifica el costo computacional adicional.
Uso del tamaño de entrada del modelo entrenado (configuración por defecto):
Utilizar el mismo tamaño de entrada que el empleado durante el entrenamiento es la configuración ideal. Esta opción preserva el contexto y las características aprendidas por el modelo, optimizando el equilibrio entre precisión y eficiencia.
4.4 Conclusión sobre el Tamaño de los Cortes
Personalización: La opción de ajustar el tamaño de los cortes está disponible para realizar pruebas o adaptarse a necesidades específicas, pero debe usarse con precaución.
Recomendación: Se recomienda utilizar el tamaño de entrada original del modelo para garantizar un rendimiento óptimo, ya que configuraciones alternativas (ya sea cortes más grandes o más pequeños) pueden no mejorar los resultados y, en muchos casos, pueden reducir la eficacia de las inferencias.
El usuario también puede decidir generar imágenes individuales de cada fragmento detectado, aunque esta funcionalidad está deshabilitada por defecto debido a limitaciones prácticas (ver Anexo 4.6).

4.3 Inferencia con SAHI
División y predicción por cortes:
Las imágenes de alta resolución se dividen en fragmentos, y cada uno es procesado individualmente.
Los resultados de los fragmentos se consolidan en una predicción global, representando todas las detecciones en la imagen original.
Registro de resultados:
Las detecciones incluyen información sobre:
Coordenadas de los bounding boxes.
Clasificación de objetos.
Niveles de confianza asociados.
Generación de imágenes estilizadas:
Los resultados visuales se presentan en un formato estilizado, mostrando etiquetas y confianza sobre la imágen completa, lo que facilita la interpretación.
4.4 Exportación de Resultados
Los resultados se exportan en formatos estandarizados para facilitar su integración y análisis:
JSON: Incluye metadatos de la imagen, coordenadas geográficas (UTM), y detalles del modelo utilizado.
CSV: Contiene un resumen de las detecciones por clase y estadísticas relevantes, como tiempo de procesamiento total y cantidad de imágenes analizadas en flujos por lotes.
Estos formatos aseguran la interoperabilidad con herramientas externas, como sistemas de información geográfica (GIS).
4.5 Georreferenciación y Análisis en QGIS
Los datos generados se integran en QGIS para la creación de mapas detallados que reflejan la distribución geográfica de las detecciones:
Se utilizan coordenadas UTM extraídas de la metadata GPS para garantizar precisión espacial.
Estos mapas proporcionan información clave para la toma de decisiones, como la densidad de malezas en áreas agrícolas.
4.6 Generación de Imágenes Individuales por Fragmento Detectado
Propósito: La funcionalidad opcional de generar imágenes individuales permite guardar los fragmentos detectados como imágenes independientes. Esto puede ser útil para:
Validación visual manual.
Creación de subconjuntos de datos para experimentos adicionales.
Almacenamiento de datos estructurados.
Consideraciones:
Por defecto desactivado:
Los objetos detectados suelen ser pequeños, lo que genera imágenes de baja resolución.
Estas imágenes pueden no ser útiles para la generación de datos sintéticos o aplicaciones adicionales.
Reducir la generación de imágenes optimiza los recursos de almacenamiento y tiempo de procesamiento.
Cuando está habilitado:
Cada fragmento detectado se recorta según las coordenadas del bounding box.
Las imágenes se guardan en un directorio organizado junto con un archivo JSON asociado que contiene:
Coordenadas de detección.
Clase del objeto.
Nivel de confianza.
4.7. Resumen Metodológico
El procedimiento con SAHI se basa en:
Preparación: Configuración de parámetros de inferencia adaptados al modelo entrenado.
Inferencia: División eficiente de imágenes, consolidación de predicciones y registro de resultados.
Exportación e integración: Generación de formatos estandarizados para análisis y representación geográfica.
Este flujo asegura precisión y eficiencia, facilitando su uso práctico en aplicaciones agrícolas de precisión. La opción de generación de imágenes individuales proporciona flexibilidad adicional para casos específicos, adaptándose a diferentes necesidades analíticas.
