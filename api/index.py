from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import plotly
import numpy as np
import plotly.graph_objs as go
import json
import requests # <--- Importa requests
import io       # <--- Importa io

app = Flask(__name__)

# --- INICIO DE CAMBIOS ---
# URLs de tus archivos subidos a Vercel Blob
# Reemplaza estas URLs con las que obtuviste
URL_MODELO = "https://gbrfrxuenvweb0kh.public.blob.vercel-storage.com/modelo_arbol.pkl"
URL_NEM = "https://gbrfrxuenvweb0kh.public.blob.vercel-storage.com/nem.csv"
URL_COMUNAS = "https://gbrfrxuenvweb0kh.public.blob.vercel-storage.com/comuna_ips.csv"
URL_COLEGIOS = "https://gbrfrxuenvweb0kh.public.blob.vercel-storage.com/colegios.csv"
URL_DATOS = "https://gbrfrxuenvweb0kh.public.blob.vercel-storage.com/datos_region_metropolitana.csv"

# Cargar modelo desde la URL
response_modelo = requests.get(URL_MODELO)
response_modelo.raise_for_status() # Lanza un error si la descarga falla
modelo = pickle.load(io.BytesIO(response_modelo.content))

# Cargar datos auxiliares desde las URLs
df_nem = pd.read_csv(URL_NEM, sep=';')
df_nem['Nota'] = df_nem['Nota'].astype(str).str.replace(',', '.').astype(float)
df_nem['Puntaje'] = df_nem['Puntaje'].astype(int)
df_nem = df_nem.sort_values('Nota')

df_comunas = pd.read_csv(URL_COMUNAS, sep=';')
df_comunas['NOMBRE_COMUNA_EGRESO'] = df_comunas['NOMBRE_COMUNA_EGRESO'].str.strip().str.upper()

df_colegios = pd.read_csv(URL_COLEGIOS, sep=';')
df_colegios['NOMBRE_COMUNA_EGRESO'] = df_colegios['NOMBRE_COMUNA_EGRESO'].str.strip().str.upper()
df_colegios['NOMBRE_UNIDAD_EDUC'] = df_colegios['NOMBRE_UNIDAD_EDUC'].str.strip().str.upper()

df_datos = pd.read_csv(URL_DATOS, sep=';')
# --- FIN DE CAMBIOS ---

# ... (El resto de tu código de Flask sigue exactamente igual)
# @app.route('/')
# def home():
# ...

# (No necesitas volver a copiar el resto del código si no ha cambiado)
@app.route('/')
def home():
    comunas = sorted(df_comunas['NOMBRE_COMUNA_EGRESO'].unique())
    return render_template('index.html', comunas=comunas, resultado=None)

@app.route('/get_colegios/<comuna>')
def get_colegios(comuna):
    comuna = comuna.strip().upper()
    colegios_filtrados = df_colegios[df_colegios['NOMBRE_COMUNA_EGRESO'] == comuna]
    lista_colegios = colegios_filtrados['NOMBRE_UNIDAD_EDUC'].drop_duplicates().tolist()
    return jsonify(lista_colegios)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    dependencia = int(data['DEPENDENCIA'])
    rama = int(data['RAMA_NUM'])
    promedio_notas = float(data['PROMEDIO_NOTAS'])
    comuna = data['COMUNA'].strip().upper()
    colegio = data['COLEGIO'].strip().upper()

    promedio_redondeado = round(promedio_notas, 2)
    if promedio_redondeado in df_nem['Nota'].values:
        ptje_nem = float(df_nem.loc[df_nem['Nota'] == promedio_redondeado, 'Puntaje'].values[0])
    else:
        ptje_nem = np.interp(promedio_notas, df_nem['Nota'], df_nem['Puntaje'])

    fila_comuna = df_comunas[df_comunas['NOMBRE_COMUNA_EGRESO'] == comuna]
    if fila_comuna.empty:
        return jsonify({"error": f"No se encontraron datos para la comuna '{comuna}'."})

    ips = float(fila_comuna.iloc[0]['IPS'])
    promedio_comuna_gral = float(fila_comuna.iloc[0]['PROMEDIO_COMUNA_gral'])
    promedio_comuna_nota = float(fila_comuna.iloc[0]['PROMEDIO_COMUNA_NOTA'])

    fila_colegio = df_colegios[
        (df_colegios['NOMBRE_COMUNA_EGRESO'] == comuna) &
        (df_colegios['NOMBRE_UNIDAD_EDUC'] == colegio)
    ]
    if fila_colegio.empty:
        return jsonify({"error": f"No se encontraron datos para el colegio '{colegio}' en la comuna '{comuna}'."})

    promedio_colegio = float(fila_colegio.iloc[0]['PROMEDIO_COLEGIO'])
    datos_colegio = df_datos[
        (df_datos["NOMBRE_COMUNA_EGRESO"] == comuna) &
        (df_datos["NOMBRE_UNIDAD_EDUC"] == colegio)
    ]["PROMEDIO_CM_MAX"].dropna()

    colegio_min = datos_colegio.min() if not datos_colegio.empty else 0
    colegio_max = datos_colegio.max() if not datos_colegio.empty else 0

    datos_modelo = pd.DataFrame([{
        'DEPENDENCIA': dependencia,
        'RAMA_NUM': rama,
        'IPS': ips,
        'PROMEDIO_NOTAS': promedio_notas,
        'PTJE_NEM': ptje_nem,
        'PROMEDIO_COMUNA_gral': promedio_comuna_gral,
        'PROMEDIO_COMUNA_NOTA': promedio_comuna_nota,
        'PROMEDIO_COLEGIO': promedio_colegio
    }])

    prediccion = modelo.predict(datos_modelo)[0]
    resultado = round(prediccion, 1)

    grafico_colegio = {
        "data": [
            go.Bar(x=["Puntaje Mínimo", "Puntaje Máximo"], y=[colegio_min, colegio_max],
                   marker_color=["#F4A261", "#2A9D8F"])
        ],
        "layout": go.Layout(
            title=f"Puntajes Históricos del Colegio {colegio}",
            xaxis=dict(title="Categoría"),
            yaxis=dict(title="Puntaje PAES"),
            height=350
        )
    }

    top_colegios = (
        df_datos[df_datos['NOMBRE_COMUNA_EGRESO'] == comuna]
        .dropna(subset=['PROMEDIO_CM_MAX'])
        .sort_values('PROMEDIO_CM_MAX', ascending=False)
        .drop_duplicates('NOMBRE_UNIDAD_EDUC')
        .head(5)
    )

    grafico_top_colegios = {
        "data": [
            go.Bar(
                x=top_colegios['NOMBRE_UNIDAD_EDUC'],
                y=top_colegios['PROMEDIO_CM_MAX'],
                marker_color="#264653"
            )
        ],
        "layout": go.Layout(
            title=f"Top 5 Colegios en {comuna} por Puntaje Máximo",
            xaxis=dict(title="Colegio", tickangle=-30),
            yaxis=dict(title="Puntaje PAES Máximo"),
            height=350
        )
    }

    return jsonify({
        "puntaje": resultado,
        "grafico_colegio": json.dumps(grafico_colegio, cls=plotly.utils.PlotlyJSONEncoder),
        "grafico_top_colegios": json.dumps(grafico_top_colegios, cls=plotly.utils.PlotlyJSONEncoder)
    })