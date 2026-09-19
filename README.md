# Proyecto de Análisis de Datos - IDEAM

Este proyecto analiza el catálogo nacional de estaciones del IDEAM utilizando Python, con el objetivo de explorar la distribución geográfica de las estaciones y evaluar la viabilidad de predecir su altitud mediante modelos de regresión.

## Estructura del Proyecto
- `data/`: Carpeta que contiene el CSV original (`proyecto.csv`) y el dataset limpio (`ideam_limpio.csv`).
- `analisis_completo.py`: Script principal con los 3 módulos del proyecto.
- `data.js`: Archivo generado automáticamente con los datos procesados para un dashboard web.

## Módulos Desarrollados
1. **Preparación de Datos:** Limpieza de coordenadas (comas a puntos), conversión de altitudes, formateo de fechas y eliminación de duplicados.
2. **Análisis Estadístico:** Cálculo de tendencia central (media, mediana, moda), dispersión (rango, varianza, desviación estándar) y detección de valores atípicos (outliers).
3. **Visualización y Modelado:** Gráficos exploratorios (barras, histogramas, boxplots, mapas de dispersión) y entrenamiento de modelos predictivos (Regresión Lineal Simple, Múltiple y Logarítmica) con validación cruzada K-Fold.

## Tecnologías Utilizadas
- Python 3.14
- Pandas, Numpy
- Matplotlib, Seaborn
- Scikit-Learn

## Cómo Ejecutar
1. Clonar este repositorio.
2. Instalar las dependencias: `pip install -r requirements.txt`
3. Ejecutar el script: `python analisis_completo.py`

## Resultados y Conclusiones
- Se analizaron **9,691 estaciones** del IDEAM.
- La altitud promedio es de **2000 msnm** con una desviación estándar de **2558 m**, lo que refleja la alta dispersión geográfica de Colombia.
- Se detectaron **932 valores atípicos** de altitud.
- Los modelos predictivos arrojaron un **R² máximo de 0.04**, lo que demuestra que las coordenadas geográficas por sí solas no son suficientes para predecir la altitud en un país con una geografía tan compleja.

## Aplicación Profesional
Este análisis permite al IDEAM identificar zonas subrepresentadas, priorizar el mantenimiento de estaciones en zonas de difícil acceso y planificar futuras instalaciones basándose en la distribución geográfica actual.