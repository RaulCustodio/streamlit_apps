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
    todos_anos = st.sidebar.checkbox('Dados de todo o período', value = True)
    if todos_anos:
         ano = ''
    else:
         ano = st.sidebar.slider('Ano', 1987, 2025)
         df = df[df['Data'].dt.year <= ano]


##Gráficos

figure_grafico_linha = px.line(df, 
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


# Visualize as previsões
fig_prophet = model_prophet.plot(forecast_prophet)
# fig_prophet.update_layout(
#     width=1000,  # Largura em pixels
#     height=500,  # Altura em pixels
# )

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
    st.plotly_chart(figure_grafico_linha)

with aba2:
    st.plotly_chart(fig_naive)

with aba3:
    st.plotly_chart(fig_prophet)

#st.table(treino)