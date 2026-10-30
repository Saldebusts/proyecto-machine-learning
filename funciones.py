"""Funciones del proyecto: cada función resuelve una tarea pequeña y explicada."""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils import resample
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier


def limpiar_columnas(datos):
    """Devuelve una copia con nombres sencillos, sin cambiar los valores."""
    datos = datos.copy()
    datos.columns = ["fallos_llamada", "quejas", "meses_suscripcion", "nivel_gasto",
                     "segundos_uso", "numero_llamadas", "numero_sms", "numeros_distintos",
                     "grupo_edad", "plan_tarifa", "estado", "edad", "valor_cliente", "abandono"]
    return datos


def preparar_variables(datos):
    """Recibe perfiles originales renombrados y devuelve las 12 entradas del modelo.

    No necesita la respuesta (abandono). Se puede usar con un cliente nuevo.
    La duración media vale cero si no hay llamadas. Los códigos binarios
    tienen significado fijo, por lo que no hace falta crear dummies.
    """
    columnas = ["fallos_llamada", "quejas", "meses_suscripcion", "nivel_gasto",
                "segundos_uso", "numero_llamadas", "numero_sms", "numeros_distintos",
                "plan_tarifa", "estado", "edad"]
    entradas = datos[columnas].copy()
    entradas["plan_tarifa"] = entradas["plan_tarifa"].map({1: 0, 2: 1})
    entradas["estado"] = entradas["estado"].map({1: 0, 2: 1})
    entradas["duracion_media_llamada"] = entradas["segundos_uso"] / entradas["numero_llamadas"].replace(0, 1)
    return entradas


def balancear(X, y, metodo="over"):
    """Balancea SOLO entrenamiento usando resample, mediante remuestreo.

    over: repite ejemplos de abandono hasta igualar las dos clases.
    under: toma menos ejemplos de permanencia, sin repetirlos.
    Se conserva la correspondencia por posición entre entradas y respuestas.
    """
    entrenamiento = X.copy()
    entrenamiento["abandono"] = y.values
    permanecen = entrenamiento[entrenamiento["abandono"] == 0]
    abandonan = entrenamiento[entrenamiento["abandono"] == 1]
    if metodo == "over":
        abandonan = resample(abandonan, replace=True, n_samples=len(permanecen), random_state=0)
    elif metodo == "under":
        permanecen = resample(permanecen, replace=False, n_samples=len(abandonan), random_state=0)
    else:
        raise ValueError("El método debe ser over o under")
    conjunto = pd.concat([permanecen, abandonan])
    return conjunto.drop(columns="abandono"), conjunto["abandono"]


def evaluar_modelo(nombre, y_real, prediccion):
    """Devuelve las cuatro métricas y los cuatro tipos de acierto/error."""
    tn, fp, fn, tp = confusion_matrix(y_real, prediccion, labels=[0, 1]).ravel()
    return {"Modelo": nombre, "Accuracy": accuracy_score(y_real, prediccion),
            "Precision": precision_score(y_real, prediccion, zero_division=0),
            "Recall": recall_score(y_real, prediccion, zero_division=0),
            "F1": f1_score(y_real, prediccion, zero_division=0),
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}


def entrenar_evaluar(nombre, modelo, X_train, y_train, X_val, y_val):
    """Entrena un modelo, predice validación y devuelve el modelo y sus métricas."""
    modelo.fit(X_train, y_train)
    prediccion = modelo.predict(X_val)
    resultado = evaluar_modelo(nombre, y_val, prediccion)
    return modelo, resultado


def crear_modelos():
    """Crea modelos nuevos para comparar entrenamiento original y balanceado.

    Pasting usa BaggingClassifier con bootstrap=False: cada muestra no repite
    filas. max_samples=0.8 permite que los árboles reciban muestras diferentes.
    """
    return {
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Regresión logística": LogisticRegression(max_iter=1000, random_state=0),
        "Árbol de decisión": DecisionTreeClassifier(max_depth=10, random_state=0),
        "Bagging": BaggingClassifier(DecisionTreeClassifier(max_depth=10), n_estimators=100, max_samples=0.8, random_state=0),
        "Pasting": BaggingClassifier(DecisionTreeClassifier(max_depth=10), n_estimators=100, max_samples=0.8, bootstrap=False, random_state=0),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=0),
        "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=0)
    }


def predecir_clientes(datos, paquete):
    """Prepara y normaliza nuevos perfiles con el escalador guardado, sin fit."""
    entradas = preparar_variables(datos)[paquete["columnas"]]
    normalizadas = pd.DataFrame(paquete["escalador"].transform(entradas),
                               columns=entradas.columns, index=entradas.index)
    salida = datos.copy()
    salida["prediccion_abandono"] = paquete["modelo"].predict(normalizadas)
    return salida


def resultados_presentacion():
    """Resultados fijos de la ejecución entregada del notebook.

    Se copian aquí para presentar sin entrenar ni añadir CSV de resultados.
    Si se repite el análisis y cambian las métricas, hay que actualizar esta tabla
    y los recuentos explicados en app.py antes de volver a presentar.
    """
    filas = [
        ['Random Forest Randomized | Original', 0.9578947368421052, 0.9452054794520548, 0.7752808988764045, 0.8518518518518519],
        ['Random Forest Grid | Original', 0.956140350877193, 0.9324324324324323, 0.7752808988764045, 0.8466257668711656],
        ['Random Forest | Original', 0.956140350877193, 0.9444444444444444, 0.7640449438202247, 0.84472049689441],
        ['Random Forest | Oversampling', 0.9473684210526316, 0.8041237113402062, 0.8764044943820225, 0.8387096774193549],
        ['Random Forest Grid | Oversampling', 0.950877192982456, 0.8674698795180723, 0.8089887640449438, 0.8372093023255814],
        ['Random Forest Randomized | Oversampling', 0.9438596491228072, 0.7878787878787878, 0.8764044943820225, 0.8297872340425532],
        ['Bagging | Oversampling', 0.9438596491228072, 0.8, 0.8539325842696629, 0.8260869565217391],
        ['Gradient Boosting | Original', 0.9491228070175438, 0.9054054054054054, 0.7528089887640449, 0.8220858895705522],
        ['Pasting | Oversampling', 0.9421052631578948, 0.8043478260869565, 0.8314606741573034, 0.8176795580110497],
        ['Gradient Boosting | Oversampling', 0.936842105263158, 0.7523809523809524, 0.8876404494382022, 0.8144329896907216],
        ['KNN | Oversampling', 0.9333333333333332, 0.7339449541284404, 0.898876404494382, 0.8080808080808081],
        ['Bagging | Original', 0.9438596491228072, 0.88, 0.7415730337078652, 0.8048780487804879],
        ['Árbol de decisión | Oversampling', 0.9280701754385964, 0.7307692307692307, 0.8539325842696629, 0.7875647668393783],
        ['Pasting | Original', 0.9350877192982456, 0.825, 0.7415730337078652, 0.7810650887573964],
        ['KNN | Original', 0.9350877192982456, 0.8333333333333334, 0.7303370786516854, 0.7784431137724551],
        ['AdaBoost | Original', 0.9315789473684212, 0.8125, 0.7303370786516854, 0.7692307692307693],
        ['Árbol de decisión | Original', 0.9263157894736842, 0.7901234567901234, 0.7191011235955056, 0.7529411764705882],
        ['AdaBoost | Oversampling', 0.8912280701754386, 0.6015037593984962, 0.898876404494382, 0.7207207207207207],
        ['Regresión logística | Undersampling', 0.856140350877193, 0.5234899328859061, 0.8764044943820225, 0.6554621848739496],
        ['Regresión logística | Oversampling', 0.8508771929824561, 0.5128205128205128, 0.898876404494382, 0.6530612244897959],
        ['Regresión logística | Original', 0.9070175438596492, 0.9285714285714286, 0.4382022471910112, 0.5954198473282443],
    ]
    return pd.DataFrame(filas, columns=["Modelo", "Accuracy", "Precision", "Recall", "F1"])


def tarjetas_presentacion(cifras, etiquetas, notas):
    """Tres cifras grandes: una figura de Matplotlib, sin HTML ni CSS."""
    fig, ejes = plt.subplots(1, 3, figsize=(10, 3))
    fig.set_facecolor("#f5f3ed")
    colores = ["#123044", "#147d82", "#bb562e"]
    for i, ax in enumerate(ejes):
        ax.set_facecolor(colores[i])
        ax.set_xticks([])
        ax.set_yticks([])
        for borde in ax.spines.values():
            borde.set_visible(False)
        ax.text(0.08, 0.78, cifras[i], fontsize=34, weight="bold", color="white", transform=ax.transAxes)
        ax.text(0.08, 0.51, etiquetas[i], fontsize=15, weight="bold", color="white", transform=ax.transAxes)
        ax.text(0.08, 0.16, notas[i], fontsize=12, color="white", transform=ax.transAxes, linespacing=1.5)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.03, wspace=0.05)
    return fig


def barras_presentacion(etiquetas, valores, titulo, unidad, colores=None):
    """Barras con valores visibles y escala desde cero para comparar sin exagerar."""
    if colores is None:
        colores = ["#147d82"] * len(valores)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.set_facecolor("#f5f3ed")
    ax.set_facecolor("#f5f3ed")
    ax.barh(etiquetas, valores, color=colores, height=0.58)
    ax.invert_yaxis()
    ax.set_title(titulo, loc="left", fontsize=16, color="#123044", pad=18)
    ax.set_xlabel(unidad, fontsize=12)
    ax.tick_params(axis="y", length=0, labelsize=12)
    ax.tick_params(axis="x", labelsize=10)
    for borde in ax.spines.values():
        borde.set_visible(False)
    for i, valor in enumerate(valores):
        texto = (f"{valor:.3f}" if unidad == "F1 · de 0 a 1" else f"{valor:g}").replace(".", ",")
        ax.text(valor, i, "  " + texto, va="center", fontsize=13, weight="bold", color="#123044")
    ax.set_xlim(0, 1.1 if unidad == "F1 · de 0 a 1" else max(valores) * 1.22)
    fig.tight_layout(pad=1.5)
    return fig


def balance_presentacion():
    """Comparación real de validación del bosque de referencia, con y sin balanceo."""
    fig, ejes = plt.subplots(1, 2, figsize=(10, 3.7))
    fig.set_facecolor("#f5f3ed")
    for ax, valores, titulo, limite in zip(ejes, [[68, 78], [4, 19]],
                                         ["Bajas detectadas · de 89", "Falsas alarmas"], [89, 89]):
        ax.set_facecolor("#f5f3ed")
        ax.bar(["Original", "Oversampling"], valores, color=["#123044", "#bb562e"], width=0.55)
        ax.set_ylim(0, limite)
        ax.set_title(titulo, loc="left", fontsize=14, pad=15)
        ax.tick_params(labelsize=12)
        for i, valor in enumerate(valores):
            ax.text(i, valor+2, str(valor), ha="center", fontsize=20, weight="bold")
        for borde in ax.spines.values():
            borde.set_visible(False)
    fig.tight_layout(pad=1.5)
    return fig


def formatear_metricas(tabla):
    """Presenta porcentajes y F1 sin cambiar la tabla utilizada para seleccionar."""
    vista = tabla.copy()
    for columna in ["Accuracy", "Precision", "Recall"]:
        if columna in vista.columns:
            vista[columna] = vista[columna].map(lambda valor: f"{valor * 100:.2f} %".replace(".", ","))
    if "F1" in vista.columns:
        vista["F1"] = vista["F1"].map(lambda valor: f"{valor:.3f}".replace(".", ","))
    return vista


def reparto_presentacion():
    """Representa el reparto proporcional y la función de cada conjunto."""
    fig, ax = plt.subplots(figsize=(10, 3.4))
    fig.set_facecolor("#f5f3ed")
    ax.set_facecolor("#f5f3ed")
    inicio = 0
    for cantidad, color, etiqueta in zip([60, 20, 20], ["#123044", "#147d82", "#bb562e"],
                                         ["60 % · 1.710", "20 % · 570", "20 % · 570"]):
        ax.barh(0.7, cantidad, left=inicio, color=color, height=0.35)
        ax.text(inicio + cantidad / 2, 0.7, etiqueta, ha="center", va="center", color="white", fontsize=16, weight="bold")
        inicio += cantidad
    for posicion, titulo, detalle in zip([30, 70, 90], ["Entrenamiento", "Validación", "Test"],
                                         ["Aprender patrones", "Elegir el modelo", "Evaluación final"]):
        ax.text(posicion, 0.35, titulo, ha="center", fontsize=14, weight="bold", color="#123044")
        ax.text(posicion, 0.17, detalle, ha="center", fontsize=11, color="#123044")
    ax.set_xlim(-1, 101)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.tight_layout(pad=1)
    return fig
