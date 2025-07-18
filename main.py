from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import plotly
import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
import json

app = Flask(__name__)

# Cargar modelo
with open('modelo_arbol.pkl', 'rb') as file:
    modelo = pickle.load(file)

# Cargar datos auxiliares
df_nem = pd.read_csv('nem.csv', sep=';')
df_nem['Nota'] = df_nem['Nota'].astype(str).str.replace(',', '.').astype(float)
df_nem['Puntaje'] = df_nem['Puntaje'].astype(int)
df_nem = df_nem.sort_values('Nota')

df_comunas = pd.read_csv('comuna_ips.csv', sep=';')
df_comunas['NOMBRE_COMUNA_EGRESO'] = df_comunas['NOMBRE_COMUNA_EGRESO'].str.strip().str.upper()

df_colegios = pd.read_csv('colegios.csv', sep=';')
df_colegios['NOMBRE_COMUNA_EGRESO'] = df_colegios['NOMBRE_COMUNA_EGRESO'].str.strip().str.upper()
df_colegios['NOMBRE_UNIDAD_EDUC'] = df_colegios['NOMBRE_UNIDAD_EDUC'].str.strip().str.upper()

df_datos = pd.read_csv('datos_region_metropolitana.csv', sep=';')

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

@app.route('/top_7_comunas')
def top_7_comunas():
    # Filtra nulos
    df_filtrado = df_datos.dropna(subset=['NOMBRE_COMUNA_EGRESO', 'PROMEDIO_CM_MAX'])

    # Agrupa por comuna y obtiene el promedio del puntaje
    top_comunas = df_filtrado.groupby('NOMBRE_COMUNA_EGRESO')['PROMEDIO_CM_MAX'].mean()

    # Ordena de mayor a menor y toma las 7 primeras
    top_comunas = top_comunas.sort_values(ascending=False).head(7).reset_index()

    # Asegura tipo de datos correcto
    top_comunas['PROMEDIO_CM_MAX'] = top_comunas['PROMEDIO_CM_MAX'].astype(float)
    top_comunas['NOMBRE_COMUNA_EGRESO'] = top_comunas['NOMBRE_COMUNA_EGRESO'].astype(str)

    # Prepara respuesta JSON
    response = {
        'comunas': top_comunas['NOMBRE_COMUNA_EGRESO'].tolist(),
        'puntajes': top_comunas['PROMEDIO_CM_MAX'].tolist()
    }

    return jsonify(response)



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
    # Valores históricos del colegio
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


    # Gráfico 1: Puntaje mínimo y máximo del colegio
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

    # Gráfico 2: Top 5 colegios de la comuna
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




if __name__ == '__main__':
    app.run(debug=True)
