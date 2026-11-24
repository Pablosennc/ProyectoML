# ProyectoML

Framework de línea de comandos para clasificar, estimar valores numéricos y
agrupar observaciones tabulares. Genera diagnósticos exploratorios, métricas,
gráficos PNG a 300 DPI y modelos serializados.

## Instalación

Se verificó con Python 3.12. Abre una terminal en la raíz del repositorio.

```bash
python -m venv .venv
```

En Windows (PowerShell), usa `.\.venv\Scripts\Activate.ps1`. En macOS o Linux,
usa `source .venv/bin/activate`. Luego instala las dependencias.

```bash
python -m pip install -r requirements.txt
python main.py --help
```

## Demostración

```bash
python main.py --data data/iris_clasificacion.csv --output-dir reports/iris
python main.py --data data/wine_sin_etiquetas.tsv --output-dir reports/wine
python main.py --demo regression --output-dir reports/regresion
```

Iris contiene 150 flores con cuatro medidas y una especie observada (UCI,
Fisher, DOI 10.24432/C56C76). Wine contiene 178 vinos con 13 medidas químicas
(UCI, Aeberhard y Forina, DOI 10.24432/C5PC7J). Los archivos proceden de los
datasets de ejemplo de scikit-learn. En Wine se omitió el cultivar.

## e-means de Meta FAIR (opcional)

K-Means sigue siendo el algoritmo predeterminado. Para probar e-means en CPU,
instala la dependencia adicional en el mismo entorno y usa la misma tabla de vinos.

```bash
python -m pip install -r requirements-emeans.txt
python main.py --data data/wine_sin_etiquetas.tsv --clusterer emeans --output-dir reports/wine-emeans
```

El motor usa `NumpyEmeans` de Faiss con cinco épocas e inicialización
propia de Faiss. Usa `--k-init` para configurar únicamente K-Means.
Ambos motores recorren el mismo rango de K y reportan inercia, silueta,
Calinski-Harabasz, Davies-Bouldin y PCA. Los CSV, modelos y figuras específicos
de e-means llevan el prefijo `emeans_`. Emplea directorios de salida separados
para conservar también el resumen JSON y EDA de cada corrida.

e-means actualiza los centroides con medias móviles exponenciales de las
estadísticas de cada minilote. La publicación evalúa las mejoras de rendimiento
principalmente con grandes colecciones de embeddings. La ejecución con Wine
sirve para comparar el resultado del agrupamiento en una tabla pequeña.

[Artículo de Meta FAIR](https://arxiv.org/abs/2611.09501) ·
[Implementación oficial en Faiss](https://github.com/facebookresearch/faiss/tree/v1.16.0/contrib/e_means)

## Datos propios

```bash
python main.py --data tabla.parquet --target resultado --task regression --models ridge lasso cart
python main.py --data tabla.tsv --mode unsupervised --k-range 2 8 --k-criterion silhouette --k-init random
```

La primera fila debe contener los nombres de las columnas. El cargador admite
CSV, TSV y Parquet, además de XLSX y XLS. En texto intenta UTF-8 y Latin-1,
entre otras codificaciones. En modo automático se toma el objetivo indicado por
`--target` o se detecta una coincidencia inequívoca por nombre entre
`target`, `label`, `class`, `clase`, `objetivo`, `etiqueta` e `y`.
Si no hay una coincidencia única, se emite un aviso y se ejecuta K-Means.
Usa `--task` cuando una variable entera con pocos niveles represente una cantidad.

## Arquitectura

- `main.py` procesa argumentos, conecta los módulos y escribe el reporte.
- `core/` contiene `DataLoader`, `AutomatedEDA` y `AdaptivePreprocessor`.
  El preprocesador aprende filtros, imputación y codificación exclusivamente
  con las filas de ajuste. Por defecto descarta IDs reconocibles, constantes y
  categóricas con más de 20 niveles. Los árboles mantienen la escala original.
- `models/` define el contrato polimórfico en `BaseModel` e implementa
  regresión logística, OLS, Ridge, Lasso, CART, Random Forest, MARS,
  GAM, KNN, DANN, LVQ1 y K-Means. MARS cuenta con implementación nativa.
  GAM usa pyGAM cuando es compatible y bases B-spline con logística
  multinomial para clasificación multiclase.
- `evaluation/` implementa validación cruzada, curvas de aprendizaje,
  métricas y gráficos. Cada fold ajusta su propio preprocesador. La selección
  usa F1 ponderado para clasificación y RMSE para regresión. El test se reserva
  para comparar el rendimiento final.

## Salidas e interpretación

Cada ejecución guarda `execution_report.json`, diagnósticos EDA y figuras.
El modo supervisado agrega comparación, búsqueda de hiperparámetros e
importancias de los árboles. En clasificación, las superficies muestran el
corte sobre los dos primeros predictores numéricos disponibles, con los demás
fijos en valores típicos. DANN y LVQ1 omiten ROC/PR si no devuelven
probabilidades. En regresión, la recta de calibración muestra un intervalo puntual de confianza
del 95 % para la media bajo los supuestos lineales. Las líneas del gráfico de
residuos indican ±1,96 desviaciones estándar como referencia descriptiva.

En agrupamiento se guardan las métricas por K y un CSV con el clúster y la
silueta individual. El codo maximiza la distancia a la recta que une los
extremos de la curva normalizada. La silueta, Calinski-Harabasz y
Davies-Bouldin ofrecen criterios complementarios.

Para reutilizar un modelo local, carga su `.pkl` con `BaseModel.load` y su
archivo `.preprocessing.pkl` con `joblib.load`. Ejecuta
`preprocessor.transform_test(datos)` antes de llamar a `model.predict`.
En clasificación, recupera los nombres con `label_encoder.inverse_transform`.
Carga únicamente archivos serializados de una fuente de confianza.

El informe técnico y la presentación están en `docs/`.
