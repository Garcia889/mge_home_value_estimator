![Magic home price predictor](/images/magic_ball.png)

Home value estimator

Este proyecto tiene como objetivo calcular el precio de una casa basado en diferentes características y factores.
El cálculo del precio se realiza utilizando un algoritmo que tiene en cuenta variables como el tamaño de la casa,
la ubicación, el número de habitaciones, entre otros.

mge_home_value_estimator  
├── README.md  
├── EDA  
│   └── EDA_Homes.ipynb  
├── data  
│   ├── data_description.txt  
│   ├── sample_submission.csv  
│   ├── test.csv  
│   └── train.csv  
├── images  
│   └── magic_ball.png  
├── logs  
│   └── training_logs.csv  
├── main_home_estimator.py  
└── src  
    ├── data_preprocessor.py  
    ├── model_selection.py  
    └── model_tuning.py  

## Dependencias

- Python (>=3.6)

## Instrucciones de Instalación

Para instalar las dependencias necesarias, ejecute el siguiente comando:

```bash
pip install -r requirements.txt
```
> **EDA:** Para explorar las variables de la fuente original de datos podemos usar el notebook:
EDA/EDA_Homes.ipynb


Instrucciones para usar la predicción de precios:
1. Ejecuta el archivo principal "calcmain_prediction.py".
2. Proporciona los datos requeridos, como el tamaño de la casa, la ubicación y el número de habitaciones.
3. El programa calculará automáticamente el precio de la casa y lo mostrará en la pantalla.

¡Disfruta calculando el precio de tu casa!

