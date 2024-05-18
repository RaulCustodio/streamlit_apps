import  streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from statsforecast import StatsForecast
from statsforecast.models import Naive, SeasonalNaive, SeasonalWindowAverage
#from statsmodels.tsa.seasonal import seasonal_decompose
#import pickle
from prophet import Prophet

st.set_page_config(layout= 'wide')

## Funções

def formata_numero(valor, prefixo = ''):
    for unidade in ['', 'mil']:
        if valor <1000:
            return f'{prefixo} {valor:.2f} {unidade}'
        valor /= 1000
    return f'{prefixo} {valor:.2f} milhões'

def wmape(y_true, y_pred):
  return np.abs(y_true - y_pred).sum()/np.abs(y_true).sum()

##Tabelas
df = pd.read_csv("./techvenv/brentdb.csv", sep=';')
df = df.rename(columns={'Preço - petróleo bruto - Brent (FOB)': 'Brent (F0B)'})
df['Data'] = pd.to_datetime(df['Data'], origin='1899-12-30', unit='D')
df['Brent (F0B)'] = df['Brent (F0B)'].str.replace(',', '.')
df['Brent (F0B)'] = df['Brent (F0B)'].str.replace(',', '.').astype(float)
df0=df

##Tabela Bolsa de Shangai

dfs1 = pd.read_csv("./techvenv/shanghai.csv", sep=',')
# Read the Shanghai data
dfs2 = pd.read_csv("./techvenv/shanghai2.csv", sep=',')
dfs = pd.concat([dfs1, dfs2], ignore_index=True)

# Convert the 'Data' column to datetime format
dfs['Data'] = pd.to_datetime(dfs['Data'])

# Replace commas with dots in each of the specified columns

dfs['Último'] = dfs['Último'].str.replace('.', '').str.replace(',', '.').astype(float)
dfs['Abertura'] = dfs['Abertura'].str.replace('.', '').str.replace(',', '.').astype(float)
dfs['Máxima'] = dfs['Máxima'].str.replace('.', '').str.replace(',', '.').astype(float)
dfs['Mínima'] = dfs['Mínima'].str.replace('.', '').str.replace(',', '.').astype(float)

## Modelo de ML Naive

treino = df.loc[df['Data']<"2024-01-18"]
valid = df.loc[(df['Data']>="2024-01-18")]
valid['unique_id'] = 1
treino['unique_id'] = 1
treino.rename(columns={'Data': 'ds'}, inplace=True)
treino.rename(columns={'Brent (F0B)': 'y'}, inplace=True) # Change the target column name to 'y'
valid.rename(columns={'Data': 'ds'}, inplace=True)
valid.rename(columns={'Brent (F0B)': 'y'}, inplace=True) 

model = StatsForecast(models=[Naive()],freq='D', n_jobs=-1)
model.fit(treino)

h = valid['ds'].nunique()
forecast_df = model.predict(h=h, level=[90])

forecast_df = forecast_df.reset_index().merge(valid, on=['ds', 'unique_id'], how='left')
forecast_df = pd.DataFrame(data=forecast_df)

## Modelo de ML Naive Bayers


## Modelo de ML Prophet

# Modelo Prophet

model_prophet = Prophet(daily_seasonality=True)  # Pode personalizar os hiperparâmetros conforme necessário
model_prophet.fit(treino)

# Crie um dataframe com as datas futuras para as quais você deseja fazer previsões
datas_futuras = model_prophet.make_future_dataframe(periods=365)  # Pode ajustar a quantidade de dias no futuro

# Faça as previsões
previsao = model_prophet.predict(datas_futuras)

# Merge dos dataframes de previsão com os valores reais usando a coluna 'ds'
forecast_prophet = pd.merge(previsao, valid, on='ds', how='left')
forecast_prophet.rename(columns={'y_y': 'y'}, inplace=True)


## Filtros

with st.sidebar:
    todos_anos = st.sidebar.checkbox('Valor do Barril Brent', value=True)
    ano = None
    if not todos_anos:
        ano = st.sidebar.slider('Ano', 1987, 2025)

    # Create a Streamlit button to toggle visibility of the second line
    show_shanghai_index = st.checkbox("Índice de Xangai")

if ano is not None:
    df0 = df[df['Data'].dt.year <= ano]
    dfs = dfs[dfs['Data'].dt.year <= ano]
##Gráficos

figure_grafico_linha = px.line(df0, 
                               x='Data', 
                               y = 'Brent (F0B)',
                               title='Preço do Barril de Petróleo BRENT')

figure_grafico_linha.update_layout(
    width=1000,  # Largura em pixels
    height=500,  # Altura em pixels
)

figure_grafico_linha.update_xaxes(showgrid=True)
figure_grafico_linha.update_yaxes(showgrid=True)


fig_naive = model.plot(treino.query('ds > "2023-01-18"'), forecast_df, level=[90], engine='plotly')
fig_naive.update_layout(
    width=1000,  # Largura em pixels
    height=500,  # Altura em pixels
)

# Plot using Plotly
fig = go.Figure()
fig.update_layout(
    width=1000,  # Largura em pixels
    height=500,  # Altura em pixels
)
# Plot 1: Preço do Petróleo Brent
fig.add_trace(go.Scatter(x=df['Data'], y=df0['Brent (F0B)'], mode='lines', name='Preço do Petróleo', line=dict(color='midnightblue')))

# Plot 2: Índice de Xangai
if show_shanghai_index:
    fig.add_trace(go.Scatter(x=dfs['Data'], y=dfs['Último'], mode='markers', name='Índice de Xangai', marker=dict(color='green', size=3), yaxis='y2'))

# Formatação do layout
fig.update_layout(title='Previsão de Preço do Petróleo e Índice de Xangai', xaxis_title='Data', legend=dict(x=0, y=1.1))

# Se o checkbox for marcado, atualize as configurações do eixo y para o Índice de Xangai
if show_shanghai_index:
    fig.update_layout(yaxis=dict(title='Preço do Petróleo (US$/barril)', color='midnightblue'),
                      yaxis2=dict(title='Índice de Xangai', color='green', overlaying='y', side='right'))


# Visualize as previsões
fig_prophet = model_prophet.plot(forecast_prophet)

##Visualização streamlit

st.title('DASHBOARD DE PREVISÃO DE PREÇO DO PETRÓLEO')
aba1, aba2, aba3 = st.tabs(['Dados Reais', 'Naive', 'Prophet'])

with aba1:
    coluna1, coluna2, coluna3 = st.columns(3)

    with coluna1:
        st.metric('Máximo',formata_numero(df['Brent (F0B)'].max(),''))
    with coluna2:
        st.metric('Mínimo',formata_numero(df['Brent (F0B)'].min(),''))
    with coluna3:
        st.metric('Média',formata_numero(df['Brent (F0B)'].mean(),''))
    # Mostrando o gráfico
    st.plotly_chart(fig)
    
    if show_shanghai_index:
        long_text = "A china é o maior comprador de petróleo do mundo e como pode ser visto em 2009 a queda do preço do barril esteve diretamente relacioada a queda da bolsa de Xangai"
    else:
        long_text = ''

    with st.container(height=300):
        st.markdown(long_text)

    st.table(dfs.head())

with aba2:
    st.plotly_chart(fig_naive)

with aba3:
    st.plotly_chart(fig_prophet)

#st.table(treino)
