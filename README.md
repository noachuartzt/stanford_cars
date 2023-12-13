# Stanford Cars Dataset
## Descripción
Este repositorio contiene el conjunto de datos Stanford Cars, que consta de imágenes tanto de entrenamiento como de prueba. Las clases están organizadas según los modelos de automóviles. El objetivo principal del proyecto es realizar la clasificación de modelos de automóviles y aplicar una técnica de autoencoder.

## Organización
### Notebooks
- `autoencoder.ipynb`: Implementación de un autoencoder que elimina los píxeles en blanco de las imágenes.
- `cnn_models_from_scratch.ipynb`: Implementación de modelos de redes neuronales convolucionales (CNN) desde cero. Incluye la creación de modelos CNN básicos.
- `transfer-learning_ResNet50.ipynb`: Implementación de transfer learning utilizando la arquitectura ResNet50.

### Models
- `*.py`: Archivos que contienen las clases de los modelos CNN implementados.

### Checkpoints
- `/checkpoints`: Carpeta que almacena los modelos guardados.
- `/transfer-learning_checkpoints`: Carpeta destinada a los puntos de control específicos para el transfer learning.

### Data
- `*.csv`: Conjuntos de datos que contienen información relevante.
- `car_data/car_data`: Imágenes de entrenamiento y prueba organizadas en carpetas correspondientes a diferentes modelos de automóviles (196 clases).

### Auxiliar Functions
- `auxiliar_functions.py`: Archivo que contiene funciones auxiliares utilizadas en los notebooks.
