# Comparación modelos de redes

Este repositorio contiene el código y los archivos asociados al artículo **"Comparación de modelos tradicionales y de machine learning para la predicción de enlaces en grafos"**. En él, se exploran y comparan distintos enfoques, desde modelos estadísticos tradicionales hasta técnicas de machine learning, evaluando su precisión y eficiencia en la predicción de enlaces en redes complejas.

## Contenido

- **Scripts y notebooks**: Implementación de los modelos y técnicas utilizadas en el estudio, incluyendo modelos estadísticos como ERGM y técnicas de machine learning como GCN y Word2Vec.
- **Resultados**: Archivos de resultados, métricas de evaluación (AUC y tiempos de ejecución), y visualizaciones que permiten analizar el rendimiento de cada modelo en diferentes redes de colaboración académica.
- **Datos**: Conjuntos de datos de redes utilizados en el análisis, como Astro-Ph, Cond-Mat, Gr-Qc, Hep-Ph, y Hep-Th.

## Requisitos

Para reproducir los experimentos, asegúrate de tener instalado:

- Python 3.8 o superior
- Librerías necesarias: `networkx`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `torch` y `tensorflow` 

## Ejecución

1. Clonar este repositorio:

   ```bash
   git clone https://github.com/damartinezsi/Comparacion-modelos-de-redes.git
   cd Comparacion-modelos-de-redes

2. Ejecutar los notebooks en el directorio notebooks para replicar los experimentos y visualizar los resultados. En caso de no querer ajustar todos los modelos, es posible cargarlos direcatamente desde la carpeta "modelos".

## Estructura del repositorio

- `Bases utilizadas/`: Conjuntos de datos de redes en formato txt. 
- `Codigo/`: Jupyter notebook y scripts con los modelos ajustados en python y R.
- `Modelos/`: Modelos GNC y Word2Vec ajustados para los diferentes conjuntos de datos.

## Contacto
Para dudas o comentarios sobre el código o el artículo, puede contactar al correo electrónico damartinezsi@unal.edu.co
