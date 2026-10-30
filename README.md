# Iranian Churn · Predicción del abandono de clientes

**Proyecto de clasificación supervisada · Salvador Cruz de Bustos · Ironhack**

Predicción de la baja de clientes de una compañía de telecomunicaciones a partir de su actividad y características del servicio. El objetivo es identificar perfiles que podrían requerir una revisión prioritaria, comparando modelos por su capacidad de detectar bajas y limitar falsas alarmas.

**Resultado principal:** el modelo seleccionado detecta **71 de las 89 bajas del test**, con **18 bajas no detectadas y 8 falsas alarmas**.

## Presentación interactiva

[Acceder a la presentación en Streamlit](https://iranian-churn-salva.streamlit.app/)

La presentación explica el contexto de los datos, su preparación, los modelos y técnicas utilizados, los resultados, las limitaciones y los próximos pasos. Incluye una demostración del modelo con perfiles ficticios.

## Datos y contexto

El conjunto **Iranian Churn**, publicado en UCI en 2020, contiene 3.150 registros, 13 variables predictoras y una respuesta binaria: `Churn`, denominada `abandono` en el análisis (**0 = permanencia; 1 = abandono**).

Según la documentación, las entradas resumen los primeros nueve meses y la respuesta se observa al terminar el mes doce. Incluyen llamadas, segundos de uso, mensajes, quejas, antigüedad, tarifa, estado y edad. El año de publicación no identifica el periodo de recogida de los datos.

No hay valores nulos. Después de conservar una copia por fila completa quedan **2.850 registros: 2.404 permanencias y 446 abandonos (15,65 %)**. El archivo original se conserva intacto.

## Metodología

1. **Revisión y limpieza:** tipos, nulos, duplicados, rangos y distribución de la respuesta. Se retiran 300 copias exactas antes de dividir.
2. **Preparación de 12 variables:** se elimina grupo de edad por redundancia y valor del cliente por desconocerse su fórmula. Se recodifican tarifa y estado, y se crea la duración media de llamada.
3. **División estratificada:** 1.710 registros para entrenamiento, 570 para validación y 570 para test, con `random_state=0`.
4. **Normalización:** MinMaxScaler ajustado solo con entrenamiento y aplicado a todos los modelos.
5. **Comparación:** ocho clasificadores, con y sin oversampling; prueba adicional de undersampling con regresión logística.
6. **Ajuste:** GridSearchCV (18 combinaciones) y RandomizedSearchCV (10 de 36 combinaciones) sobre Random Forest, con cinco divisiones internas y F1.
7. **Selección:** mejor F1 de validación entre 21 configuraciones; recall resuelve empates. El test queda reservado para la evaluación final.
8. **Persistencia:** almacenamiento del modelo, escalador y columnas para predecir nuevos perfiles sin entrenar de nuevo.

| Modelos de referencia | Enfoque |
|---|---|
| KNN | Similitud entre perfiles |
| Regresión logística | Referencia lineal para clasificación |
| Árbol de decisión | Reglas de separación |
| Bagging y Pasting | Combinación de árboles con distintas muestras |
| Random Forest | Bosque con aleatoriedad en muestras y variables |
| Gradient Boosting y AdaBoost | Combinación secuencial de modelos |

## Resultados

Se selecciona **Random Forest con Randomized Search, 200 árboles y profundidad máxima de 10, sin balanceo**, con entradas normalizadas. Obtiene **F1 = 0,852 en validación**.

| Métrica de test | Resultado | Interpretación |
|---|---:|---|
| Accuracy | 95,44 % | 544 registros clasificados correctamente de 570 |
| Precision | 89,87 % | 71 bajas reales entre 79 alertas |
| Recall | 79,78 % | 71 bajas detectadas de 89 reales |
| F1 | 0,845 sobre 1 | Equilibrio entre precision y recall |

La referencia de predecir siempre permanencia obtiene 84,39 % de accuracy en test, pero no detecta ninguna baja.

### Hallazgos

- **Balancear cambia el tipo de error.** En el bosque de referencia, el oversampling aumenta las bajas detectadas en validación de 68 a 78 y las falsas alarmas de 4 a 19. F1 disminuye de 0,845 a 0,839.
- **El ajuste aporta una mejora pequeña.** El F1 de validación pasa de 0,845 en el bosque de referencia a 0,852 en el seleccionado.
- **Quejas, uso y estado aportan información.** Son las tres primeras variables por importancia en el bosque de referencia. Eliminar estado reduce su F1 de validación de 0,845 a 0,758. La importancia no implica causalidad.

## Estructura del repositorio

| Ruta | Contenido |
|---|---|
| `analisis_churn.ipynb` | Análisis, decisiones, código, gráficos y resultados ejecutados |
| `funciones.py` | Preparación, balanceo, evaluación, predicción y visualización |
| `app.py` | Presentación interactiva y demostración en Streamlit |
| `data/Customer_Churn.csv` | Dataset original |
| `modelos/modelo_churn.joblib` | Modelo, escalador y orden de columnas |
| `requirements.txt` | Dependencias del proyecto |
| `.gitignore` | Exclusión de cachés y archivos auxiliares |

Las transformaciones se realizan en memoria. Las tablas y gráficos permanecen en el notebook; no se generan carpetas adicionales de resultados. Las cifras de Streamlit corresponden a la ejecución entregada: ante un nuevo entrenamiento deben actualizarse junto al modelo.

## Ejecución

Entorno utilizado: **Python 3.12 y scikit-learn 1.8.0**. Desde la raíz del repositorio:

```bash
conda create -n churn python=3.12
conda activate churn
python -m pip install -r requirements.txt
```

Para reproducir el análisis, abrir `analisis_churn.ipynb`, seleccionar el entorno `churn`, reiniciar el kernel y ejecutar las celdas en orden. El notebook vuelve a guardar el modelo al finalizar.

Para abrir la presentación:

```bash
python -m streamlit run app.py
```

La aplicación utiliza el modelo guardado. Su recorrido incluye contexto, datos, preparación, modelos, técnicas, resultados, insights, demostración, dificultades y próximos pasos. Los perfiles de la demostración son ficticios y no se utilizan para evaluar calidad.

La dirección local no equivale a un despliegue público. La ejecución debe iniciarse desde la carpeta que contiene `app.py`.

### Compatibilidad en Windows

En un entorno Windows con Control inteligente de aplicaciones, la distribución de SciPy instalada inicialmente mediante pip fue bloqueada por la política de firma. La incidencia se resolvió sustituyendo SciPy y NumPy por los paquetes del canal `defaults`, sin desactivar la protección:

```bash
conda install --channel defaults --force-reinstall scipy numpy=2.3.5
python -c "import scipy.linalg; import sklearn; print('Importaciones correctas')"
```

Este paso documenta la incidencia observada; no es necesario en instalaciones que ya funcionan. El modelo requiere las versiones compatibles indicadas en las dependencias.

## Limitaciones

- Datos históricos de una sola compañía, sin identificadores ni fechas individuales. No se puede garantizar independencia por cliente ni generalización temporal.
- Las copias exactas no necesariamente son errores. Retirarlas modifica el peso de los perfiles. Se conservan entradas iguales con respuestas distintas.
- La edad tiene cinco valores y existen niveles de gasto fuera del rango descrito por la fuente; no se corrigen sin evidencia.
- El F1 de entrenamiento (0,974) supera al de validación (0,852), señal de posible sobreajuste.
- El escalado previo a la validación cruzada utiliza todo entrenamiento, siguiendo el procedimiento del curso. Las búsquedas se limitan a Random Forest; el test permanece fuera del ajuste y de la selección. No se balancean datos antes de las divisiones internas de las búsquedas.
- La disponibilidad de estado antes de una baja requiere verificación. Las importancias de árboles pueden verse afectadas por correlaciones y número de valores distintos.
- No se ha medido ahorro ni retención conseguida mediante una intervención. El resultado es educativo y no constituye validación de uso empresarial.

## Líneas de continuidad

Validar con datos recientes y separación temporal, comprobar la disponibilidad de las entradas, estimar costes de falsas alarmas y bajas no detectadas, y evaluar una intervención limitada con supervisión humana.

## Fuente y licencia de los datos

Iranian Churn [Dataset]. (2020). UCI Machine Learning Repository. [https://doi.org/10.24432/C5JW3Z](https://doi.org/10.24432/C5JW3Z).

Datos distribuidos bajo [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/). La licencia citada corresponde al dataset.
