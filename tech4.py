import  streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
#from statsmodels.tsa.seasonal import seasonal_decompose
#mport pickle

## Tabelas

df = pd.read_csv("brentdb.csv", sep=';')
df = df.rename(columns={'Preço - petróleo bruto - Brent (FOB)': 'Brent (F0B)'})
df['Data'] = pd.to_datetime(df['Data'], origin='1899-12-30', unit='D')
df['Brent (F0B)'] = df['Brent (F0B)'].str.replace(',', '.')
df['Brent (F0B)'] = df['Brent (F0B)'].str.replace(',', '.').astype(float)


figure_grafico_linha = px.line(df, 
                               x='Data', 
                               y = 'Brent (F0B)')
#figure_grafico_linha.update_layout(grid=True)

##Visualização streamlit

st.title('DASHBOARD DE PREVISÃO DE PREÇO DO PETRÓLEO')

# Mostrando o gráfico
st.plotly_chart(figure_grafico_linha)
