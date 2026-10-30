"""Presentación visual. Ejecutar desde esta carpeta: python -m streamlit run app.py."""

import pandas as pd
import matplotlib.pyplot as plt
import joblib
import streamlit as st
from funciones import (limpiar_columnas, predecir_clientes, resultados_presentacion,
                       tarjetas_presentacion, barras_presentacion, balance_presentacion,
                       formatear_metricas, reparto_presentacion)


def mostrar(figura):
    """Muestra la figura y libera memoria al cambiar de apartado."""
    st.pyplot(figura)
    plt.close(figura)


def main():
    st.write("**IRANIAN CHURN · SALVADOR CRUZ DE BUSTOS · IRONHACK**")
    apartados = ["01 · Contexto y objetivo", "02 · Los datos", "03 · Preparación",
                 "04 · Modelos y técnicas", "05 · Balanceo y ajuste", "06 · Resultado final",
                 "07 · Insights", "08 · Demostración", "09 · Problemas encontrados", "10 · Próximos pasos"]
    pagina = st.selectbox("Recorrido de la presentación", apartados)
    numero = apartados.index(pagina) + 1
    st.write(f"**{numero:02d} / 10**")

    if numero == 1:
        st.title("Anticipar la baja. Revisar el servicio a tiempo.")
        st.write("**Churn significa abandono de clientes.** El proyecto utiliza datos históricos de una compañía de telecomunicaciones iraní para predecir dos respuestas: permanece o abandona.")
        mostrar(tarjetas_presentacion(["9 meses", "3 meses", "Mes 12"],
            ["Actividad observada", "Margen temporal", "Respuesta"],
            ["Uso, quejas y servicio", "Entre los datos\ny la respuesta", "Permanencia o abandono\nregistrado al final"]))
        st.write("### ¿Qué perfiles requieren una revisión prioritaria del servicio?")
        st.write("La utilidad propuesta es priorizar atención. La capacidad de evitar bajas tendría que medirse en una prueba posterior.")
        st.write("Fuente: [Iranian Churn · UCI · 2020](https://doi.org/10.24432/C5JW3Z) · CC BY 4.0. Datos históricos; 2020 es el año de publicación del dataset.")

    elif numero == 2:
        st.title("Solo el 15,65 % abandona")
        mostrar(tarjetas_presentacion(["3.150", "300", "2.850"],
            ["Registros originales", "Copias exactas", "Para el análisis"],
            ["13 entradas + respuesta", "Retiradas antes\nde dividir los datos", "446 bajas\n2.404 permanencias"]))
        st.write("**Predecir siempre permanencia acertaría el 84,35 % del conjunto limpio, sin detectar ninguna baja.**")
        st.write("Cada fila representa un perfil de cliente: llamadas, SMS, quejas, antigüedad, tarifa, estado y edad. No hay valores ausentes.")
        if st.checkbox("Ver la fuente y la muestra"):
            datos = limpiar_columnas(pd.read_csv("data/Customer_Churn.csv"))
            st.dataframe(datos.head())
            st.write("Sin identificador, no es posible confirmar si las copias exactas son errores. Eliminarlas cambia el peso de algunos perfiles. Se conserva intacto el CSV original.")
            st.write("[Fuente y documentación · UCI](https://doi.org/10.24432/C5JW3Z)")

    elif numero == 3:
        st.title("Aprender, elegir y evaluar por separado")
        mostrar(reparto_presentacion())
        st.write("**12 entradas para predecir:** se eliminan grupo de edad y valor del cliente, se recodifican tarifa y estado y se crea duración media de llamada.")
        st.write("**MinMaxScaler en todos los modelos:** aprende los rangos solo con entrenamiento. La división conserva aproximadamente la proporción de bajas.")
        if st.checkbox("Detalle de la preparación"):
            st.table(pd.DataFrame({"Decisión": ["Quitar grupo de edad", "Quitar valor del cliente", "Tarifa y estado", "Duración media"],
                "Motivo": ["Repite información de edad", "Fórmula no disponible", "Convertir códigos 1/2 en 0/1", "Segundos divididos entre llamadas; cero si no hay llamadas"]}))
            st.write("La respuesta abandono se separa de las entradas. Normalizar y balancear nunca utiliza el test para aprender.")

    elif numero == 4:
        st.title("Ocho modelos. La misma comparación.")
        resultados = resultados_presentacion()
        base = resultados[resultados["Modelo"].str.endswith(" | Original") & ~resultados["Modelo"].str.contains("Grid|Randomized")].sort_values("F1", ascending=False)
        mostrar(barras_presentacion(base["Modelo"].str.replace(" | Original", "", regex=False).tolist(), base["F1"].tolist(),
                                   "Modelos de referencia · validación", "F1 · de 0 a 1"))
        st.write("**F1 · cuanto más cerca de 1, mejor.** Resume el equilibrio entre detectar bajas y acertar las alertas. No es un porcentaje de clientes acertados.")
        if st.checkbox("Modelos y técnicas: explicación sencilla"):
            st.write("KNN compara perfiles parecidos; regresión logística aporta una referencia sencilla; el árbol aprende reglas. Bagging, Pasting, Random Forest, Gradient Boosting y AdaBoost combinan modelos. Se comparan los ocho clasificadores sobre los mismos datos.")
            st.write("Técnicas: normalización, creación y selección de variables, oversampling, undersampling, validación cruzada de cinco partes, Grid Search y Randomized Search.")

    elif numero == 5:
        st.title("Detectar más bajas tiene un coste")
        mostrar(balance_presentacion())
        st.write("**Random Forest de referencia · validación:** el balanceo detectó 10 bajas más, pero añadió 15 falsas alarmas. F1: **0,845 → 0,839**.")
        st.write("**El ajuste mejoró poco:** Randomized Search elevó el F1 de 0,845 a **0,852**, sin balancear. Se selecciona Random Forest ajustado: 200 árboles, profundidad máxima 10.")
        if st.checkbox("Ver las búsquedas y el balanceo"):
            st.write("Oversampling: repetir bajas solo en entrenamiento hasta igualar ambas clases, probado con los ocho modelos. Undersampling: reducir permanencias, probado en regresión logística.")
            st.write("Grid Search: 18 combinaciones. Randomized Search: 10 combinaciones. Ambas usan cinco divisiones internas de entrenamiento y F1. Ganador: 200 árboles, profundidad máxima 10.")
            st.write("El escalado se ajustó con todo entrenamiento antes de la validación cruzada. Las búsquedas se limitaron a Random Forest y no usaron datos balanceados antes de sus divisiones internas.")
            st.dataframe(formatear_metricas(resultados_presentacion()))

    elif numero == 6:
        st.title("71 de 89 bajas detectadas en el test")
        mostrar(tarjetas_presentacion(["71", "18", "8"],
            ["Bajas detectadas", "Bajas no detectadas", "Falsas alarmas"],
            ["Clientes que abandonan\ny generan alerta", "Clientes que abandonan\nsin generar alerta", "Clientes que permanecen\ny generan alerta"]))
        st.write("**Random Forest ajustado · 200 árboles · profundidad 10 · sin balanceo**")
        st.write("### Detección de bajas: 79,78 % · Acierto de alertas: 89,87 %")
        st.write("**F1: 0,845 sobre 1.** Acierto total (accuracy): **95,44 %**. La referencia «siempre permanece» acierta el 84,39 %, pero no detecta bajas.")
        if st.checkbox("Cómo leer estas métricas"):
            st.write("Recall: 71 de 89 bajas reales detectadas. Precision: de las 79 alertas, 71 eran bajas reales. F1 resume el equilibrio. El modelo se eligió antes, usando validación; este test se usó al final.")

    elif numero == 7:
        st.title("Las quejas destacan; no explican todo")
        mostrar(barras_presentacion(["Quejas", "Segundos de uso", "Estado", "Número de llamadas", "Duración media"],
            [19.67, 12.56, 12.38, 11.00, 11.00], "Cinco principales variables · Random Forest de referencia", "Importancia del modelo (%)"))
        st.write("**Insight 1:** las quejas y el uso ayudan a distinguir perfiles. La importancia del modelo no demuestra causalidad.")
        st.write("**Insight 2:** sin estado, el F1 de validación baja de **0,845 a 0,758**. Es necesario verificar que ese dato se conoce antes de predecir.")
        if st.checkbox("Aplicación posible del hallazgo"):
            st.write("Priorizar una revisión del servicio y comprobar problemas de uso. Antes de intervenir, habría que medir el coste de las falsas alarmas y si esa actuación evita bajas. Las importancias pertenecen al bosque de referencia; no son probabilidades ni efectos causales.")

    elif numero == 8:
        st.title("Un perfil. Una predicción.")
        st.write("Ejemplos ficticios: muestran el funcionamiento, no miden la calidad del modelo.")
        perfiles = pd.DataFrame({
            "fallos_llamada": [2, 18, 5], "quejas": [0, 1, 0], "meses_suscripcion": [36, 15, 30],
            "nivel_gasto": [2, 0, 1], "segundos_uso": [6000, 400, 2500],
            "numero_llamadas": [100, 10, 50], "numero_sms": [80, 2, 20],
            "numeros_distintos": [35, 5, 20], "plan_tarifa": [1, 1, 2],
            "estado": [1, 2, 1], "edad": [30, 25, 45]})
        opciones = ["A · Uso frecuente", "B · Poco uso", "C · Uso intermedio"]
        elegido = st.selectbox("Perfil de ejemplo", opciones)
        cliente = perfiles.iloc[[opciones.index(elegido)]].copy()
        # Dos controles permiten explorar el perfil sin alargar la demostración.
        cliente["quejas"] = int(st.checkbox("El cliente ha presentado quejas", value=bool(cliente["quejas"].iloc[0]), key="quejas_" + elegido))
        cliente["meses_suscripcion"] = st.slider("Antigüedad del cliente (meses)", 3, 47, int(cliente["meses_suscripcion"].iloc[0]), key="meses_" + elegido)
        st.write("El resto de los valores permanece fijo en cada perfil. Cambiar una entrada puede cambiar la predicción, pero no demuestra una relación de causa y efecto.")
        st.pyplot(tarjetas_presentacion(
            [str(int(cliente["segundos_uso"].iloc[0])), str(int(cliente["numero_llamadas"].iloc[0])), "Sí" if int(cliente["quejas"].iloc[0]) else "No"],
            ["Segundos de uso", "Llamadas", "Quejas"],
            ["Actividad acumulada", "Perfil ficticio", "Dato editable"]))
        plt.close("all")
        if st.checkbox("Ver todos los datos del ejemplo"):
            ficha = cliente.T.reset_index()
            ficha.columns = ["Variable", "Valor"]
            st.table(ficha)
            st.write("Códigos: tarifa 1 = prepago y 2 = contrato; estado 1 = activo y 2 = inactivo; quejas 0 = no y 1 = sí. Nivel de gasto es una categoría, no euros.")
        if st.button("Obtener predicción"):
            paquete = joblib.load("modelos/modelo_churn.joblib")
            salida = predecir_clientes(cliente, paquete)
            if salida["prediccion_abandono"].iloc[0] == 1:
                st.write("### Predicción: posible abandono")
                st.write("Este perfil podría priorizarse para revisar el servicio. Es una estimación, no una certeza.")
            else:
                st.write("### Predicción: permanencia")
                st.write("El modelo no genera una alerta para este perfil. Eso no garantiza que el cliente permanezca.")
            st.write("Se utilizan el modelo y el escalador guardados, con la misma preparación del notebook.")

    elif numero == 9:
        st.title("Tres dificultades y sus implicaciones")
        mostrar(tarjetas_presentacion(["15,65 %", "300", "0,974"],
            ["Clase minoritaria", "Copias exactas", "F1 de entrenamiento"],
            ["Comparación de balanceo\ny evaluación con F1", "Una copia por registro\nantes de dividir", "Frente a 0,852 en validación:\nseñal de sobreajuste"]))
        st.write("**Límites de los datos:** una compañía, muestra histórica, sin fechas individuales ni identificadores. El resultado no acredita el rendimiento con clientes futuros.")
        if st.checkbox("Problema técnico y otras limitaciones"):
            st.write("En Windows, Python 3.14 no encajaba con las versiones fijadas. Se creó un entorno con Python 3.12. El Control de aplicaciones bloqueó una biblioteca de SciPy; instalarla desde Anaconda permitió cargarla manteniendo la protección activada.")
            st.write("La edad solo toma cinco valores. Se conservaron perfiles con respuestas distintas y niveles de gasto fuera del rango descrito por la fuente. No se corrigieron valores sin evidencia suficiente.")

    else:
        st.title("El siguiente paso: comprobar la utilidad real")
        mostrar(tarjetas_presentacion(["01", "02", "03"],
            ["Validar", "Medir", "Probar"],
            ["Nuevos clientes y fechas\nDisponibilidad de estado", "Coste de cada error\nCapacidad de atención", "Intervención pequeña\nRetención conseguida"]))
        st.write("### La conclusión")
        st.write("**El modelo detecta aproximadamente 8 de cada 10 bajas en el test.** Es un apoyo para priorizar revisiones, no una garantía ni una demostración de ahorro.")
        st.write("**Gracias · ¿Preguntas?**")


if __name__ == "__main__":
    main()
