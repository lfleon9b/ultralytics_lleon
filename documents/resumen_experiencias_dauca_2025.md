# Resumen de Experiencias: Detección Específica de Daucus carota (Santa Rosa, Dic 2025)

## Contexto
En paralelo al desarrollo de modelos multi-cultivo, se ejecutó una validación específica para la detección de *Daucus carota* (DAUCA) en el sector Santa Rosa. Este esfuerzo buscó evaluar la capacidad de la arquitectura YOLO11l para especializarse en una única clase de maleza bajo condiciones de alta resolución (1024px), simulando un escenario de control focalizado o "spot spraying".

## Metodología Técnica
Se aplicó el mismo estándar metodológico **DLM2** utilizado en los experimentos mayores, garantizando consistencia técnica:
*   **Dataset**: `sr_dauca` (Santa Rosa Dauca).
*   **Arquitectura**: YOLO11l (Large), seleccionada por su equilibrio entre capacidad de extracción de características y velocidad de inferencia.
*   **Entrenamiento**: Ejecutado en infraestructura dual NVIDIA RTX 4090 con estrategia DDP (Distributed Data Parallel), optimizador AdamW y 50 épocas de convergencia.
*   **Resolución**: 1024x1024 píxeles, sin aumentación interna durante el entrenamiento para preservar la fidelidad radiométrica de las imágenes pre-procesadas.

## Resultados Cuantitativos
El modelo especializado demostró un rendimiento sobresaliente en la identificación de *Daucus carota*, validando la viabilidad técnica de soluciones dedicadas para especies problemáticas.

*   **Precisión Operacional**: Se alcanzó una **Precisión del 84.6%**, lo que indica una alta selectividad del modelo. Esto es crucial en escenarios de aplicación de herbicidas, ya que minimiza los "falsos positivos" (rociar donde no hay maleza), generando ahorro directo de insumos.
*   **Detectabilidad (Recall)**: El modelo logró recuperar el **79.1%** de las instancias presentes, asegurando que la gran mayoría de los focos de infestación sean tratados.
*   **Robustez General**: El indicador **mAP50 se situó en 86.4%**, confirmando que la red neuronal puede localizar y clasificar correctamente la maleza con un solapamiento significativo sobre el objeto real.
*   **Estabilidad**: Las métricas obtenidas en los conjuntos de validación y prueba fueron consistentes (mAP50 de 80.5% vs 86.4%), lo que sugiere que el modelo aprendió características generalizables de la especie y no memorizó simplemente los datos de entrenamiento.

## Conclusión Técnica
Esta experiencia confirma que la arquitectura YOLO11l es altamente efectiva para tareas de detección mono-específica en agricultura de precisión. La capacidad del sistema para alcanzar métricas sobre el 80% en Precisión y mAP50 habilita su uso inmediato para la generación de mapas de prescripción en sitios con alta prevalencia de *Daucus carota*, complementando la estrategia general de manejo integrado de malezas del proyecto.

## Documentación de Respaldo
*   *Registro de experimento*: `/home/malezainia1/dev/ultralytics/experiments/sr_dauca/experiment_log.md`
*   *Reporte de métricas*: `/home/malezainia1/dev/ultralytics/experiments/sr_dauca/final_evaluation_report.csv`

