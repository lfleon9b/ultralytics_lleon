# Resumen Extendido de Experiencias: Detección Multi-Cultivo (Junio-Diciembre 2025)

## Contexto
Durante el segundo semestre de 2025, el proyecto consolidó su estrategia de detección de malezas evolucionando desde modelos específicos por cultivo hacia una arquitectura unificada y escalable. Se procesó y entrenó el conjunto de datos `merge_varios_cultivos` (ubicado en `/home/malezainia1/dev/ultralytics/datasets/merge_varios_cultivos`), el cual representa el esfuerzo de integración más ambicioso hasta la fecha.

## Metodología Técnica
Se implementó un flujo de entrenamiento alineado con la metodología **DLM2**, utilizando la arquitectura **YOLO11l (Large)** sobre una infraestructura de doble GPU NVIDIA RTX 4090.
*   **Dataset**: 5,765 imágenes de entrenamiento, 181 de validación y 140 de test.
*   **Resolución**: 1024x1024 píxeles para capturar detalles finos de malezas en estados fenológicos tempranos.
*   **Configuración**: 50 épocas, optimizador AdamW (Auto), Batch Size=10 (DDP), sin aumentación interna (pre-aumentado).

## Resultados Cuantitativos (Modelo Unificado)
El modelo unificado demostró una robustez excepcional al integrar seis clases taxonómicas complejas (AMBEL, LENCU, LOLSS, POLAV, POLPE, RAPRA) en un solo motor de inferencia.

*   **Rendimiento Global**: El modelo alcanzó un **mAP50 de 81.9%** y una **Precisión global de 79.9%** en el conjunto de prueba independiente. Esto valida la hipótesis de que un solo modelo "Large" puede manejar la variabilidad fenotípica de múltiples especies simultáneamente sin degradación significativa del rendimiento.
*   **Protección del Cultivo (LENCU)**: La clase Lenteja (LENCU) obtuvo un rendimiento sobresaliente con un **F1-Score de 0.90** y un **mAP50 de 94.3%**. Este resultado es crítico para la seguridad operacional de las aplicaciones de tasa variable (VRA), garantizando que el sistema distingue con alta fiabilidad el cultivo de la maleza, evitando la aspersión accidental sobre las plantas de interés comercial.
*   **Desempeño por Especie**:
    *   **Alta Detectabilidad**: *Ambrosia artemisiifolia* (AMBEL) y *Raphanus raphanistrum* (RAPRA) mostraron una detectabilidad superior, con mAP50 de **89.1%** y **87.6%** respectivamente.
    *   **Desafíos Persistentes**: *Polygonum aviculare* (POLAV) se mantiene como la clase más desafiante con un **mAP50 de 57.0%**. Este menor rendimiento se atribuye al desbalance de clases en el dataset histórico y a la similitud morfológica con otras coberturas vegetales en estadios tempranos. Actualmente, se está implementando una estrategia de corrección mediante recolección dirigida de datos para esta especie.

## Impacto Operacional y Escalabilidad
La transición a este modelo unificado simplifica drásticamente la logística de despliegue en campo. En lugar de gestionar múltiples pesos neuronales por predio, un solo archivo (`yolo11l.pt`) puede ser desplegado en drones o maquinaria terrestre para operar en rotaciones de trigo y leguminosas.

## Integración Multi-Escala
Si bien el foco de este reporte es la detección, estos resultados alimentan directamente el sistema integrado multi-escala. Las detecciones del modelo YOLO11l sirven como la capa de "verdad terrestre" de alta resolución (1.5mm/px) que, al cruzarse con las anomalías de NDVI de Sentinel-2, permiten validar las zonas de manejo. Análisis geoestadísticos preliminares sobre estos mapas de densidad han revelado una autocorrelación espacial significativa (Índice de Moran I = 0.667), confirmando que las malezas se agregan en parches manejables y no aleatorios, habilitando una reducción potencial de agroquímicos del 35-50%.

## Documentación de Respaldo
Los registros detallados de entrenamiento, curvas de convergencia, matrices de confusión y reportes métricos clase por clase se encuentran documentados en el **Anexo 1: Modelos de detección multi-cultivo y sistema integrado multi-escala**.

*   *Ruta de evidencia técnica*: `/home/malezainia1/dev/ultralytics/experiments/merge_varios_cultivos/experiment_log.md`
*   *Archivo de métricas*: `/home/malezainia1/dev/ultralytics/experiments/merge_varios_cultivos/final_evaluation_report.csv`

