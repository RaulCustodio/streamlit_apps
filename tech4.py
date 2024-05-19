import  streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from statsforecast import StatsForecast
from statsforecast.models import Naive, SeasonalNaive, SeasonalWindowAverage
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
#import pickle
from prophet import Prophet
import openpyxl

st.set_page_config(layout= 'wide')

##Tabelas
df = pd.read_csv("./techvenv/brentdb.csv", sep=';')
df = df.rename(columns={'Preço - petróleo bruto - Brent (FOB)': 'Brent (F0B)'})
df['Data'] = pd.to_datetime(df['Data'], origin='1899-12-30', unit='D')
df['Brent (F0B)'] = df['Brent (F0B)'].str.replace(',', '.')
df['Brent (F0B)'] = df['Brent (F0B)'].str.replace(',', '.').astype(float)
df0=df

# Calcular desvio padrão móvel
window = 30  # Janela de 30 dias
#df0['DesvioPadrao'] = df0['Brent (F0B)'].rolling(window=window).std()
#df0['MM30'] = df0['Brent (F0B)'].rolling(30).mean().shift() #média móvel
#df0['MM180'] = df0['Brent (F0B)'].rolling(180).mean().shift() #média móvel

# Caminho do arquivo Excel
caminho_arquivo = 'iea.xlsx'
nome_aba = 'TimeSeries_1971-2022'

# Ler o arquivo Excel e acessar a aba TimeSeries a partir da linha 2
#dados = pd.read_excel("./techvenv\iea.xlsx", sheet_name='TimeSeries_1971-2022', header=1)

try:
 dados = pd.read_excel(caminho_arquivo, sheet_name=nome_aba, header=1)
 dados = dados.rename(columns={'2022 Provisional': '2022'})
 dados.drop(columns=['NoCountry', 'NoProduct', 'NoFlow'], inplace=True)

except FileNotFoundError:
 st.error("Arquivo não encontrado.")
except Exception as e:
 st.error("Ocorreu um erro ao ler o arquivo:")

##Tabela Bolsa de Shangai

dfs1 = pd.read_csv("shanghai.csv", sep=',')
# Read the Shanghai data
dfs2 = pd.read_csv("shanghai2.csv", sep=',')
dfs = pd.concat([dfs1, dfs2], ignore_index=True)

# Convertendo a coluna 'Data' para formato datetime
dfs['Data'] = pd.to_datetime(dfs['Data'])

# Substituindo vírgulas por pontos nas colunas específicas
dfs['Último'] = dfs['Último'].str.replace('.', '').str.replace(',', '.').astype(float)
dfs['Abertura'] = dfs['Abertura'].str.replace('.', '').str.replace(',', '.').astype(float)
dfs['Máxima'] = dfs['Máxima'].str.replace('.', '').str.replace(',', '.').astype(float)
dfs['Mínima'] = dfs['Mínima'].str.replace('.', '').str.replace(',', '.').astype(float)

## Funções
def formata_numero(valor, prefixo = ''):
    for unidade in ['', 'mil']:
        if valor <1000:
            return f'{prefixo} {valor:.2f} {unidade}'
        valor /= 1000
    return f'{prefixo} {valor:.2f} milhões'

def wmape(y_true, y_pred):
  return np.abs(y_true - y_pred).sum()/np.abs(y_true).sum()

def plotagem(dados2):
    # Filtra os dados pelo país "World" e o fluxo "Total final consumption (PJ)"
    dadosf = dados2[(dados2["Country"] == "World") & (dados2["Flow"] == "Total final consumption (PJ)")]

    # Cria uma figura vazia
    fig = go.Figure()

    # Itera sobre cada produto único e adiciona a linha correspondente ao gráfico
    for product in dadosf["Product"].unique():
        dados_produto = dadosf[dadosf["Product"] == product]
        d2 = dados_produto.iloc[:, 3::].T.reset_index()
        
        if d2.shape[1] == 2:  # Verifica se há exatamente 2 colunas
            d2.columns = ["Ano", "Valor"]
            
            # Adiciona a linha ao gráfico existente
            fig.add_trace(go.Scatter(x=d2["Ano"], y=d2["Valor"], mode='lines', name=product))

    # Atualiza o título e adiciona a legenda
    fig.update_layout(title="Consumo Energético Global por Produto (KTOE)",
                      xaxis_title="Ano",
                      yaxis_title="Valor",
                      legend_title="Produto",
                      template="plotly",
                      width=600,
                      height=500)

    return fig

## Modelo de ML Naive
df['unique_id'] = 1

treino = df.loc[df['Data']<"2024-01-18"].rename(columns={'Data': 'ds', 'Brent (F0B)': 'y'})
valid = df.loc[(df['Data']>="2024-01-18")].rename(columns={'Data': 'ds', 'Brent (F0B)': 'y'})

model = StatsForecast(models=[Naive()],freq='D', n_jobs=-1)
model.fit(treino)

h = valid['ds'].nunique()
forecast_df = model.predict(h=h, level=[90])
forecast_df = forecast_df.reset_index().merge(valid, on=['ds', 'unique_id'], how='left')
#forecast_df = pd.DataFrame(data=forecast_df)

fig_naive = model.plot(treino.query('ds > "2023-01-18"'), forecast_df, level=['90'], engine='plotly')
fig_naive.update_layout(
    width=1000,  # Largura em pixels
    height=500,  # Altura em pixels
)

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

forecast_prophet2 = forecast_prophet.reset_index().merge(valid, on=['ds', 'unique_id'], how='left')
forecast_prophet2 = pd.DataFrame(data=forecast_prophet2)
forecast_prophet2 = forecast_prophet2.dropna()

fig_prophet = model_prophet.plot(forecast_prophet)

## Filtros

with st.sidebar:
 
    data_inicial_padrao = df['Data'].min()
    data_final_padrao = df['Data'].max()

    data_inicial = st.date_input("Data Inicial", value=data_inicial_padrao, min_value=df['Data'].min())
    data_final = st.date_input("Data Final", value=data_final_padrao, max_value=df['Data'].max())

# Create a Streamlit button to toggle visibility of the second line
with st.sidebar:
    show_shanghai_index = st.checkbox("Bolsa de Xangai")

df['ds'] = pd.to_datetime(df['Data'])
dfs['ds'] = pd.to_datetime(dfs['Data'])
data_inicial = pd.to_datetime(data_inicial)
data_final = pd.to_datetime(data_final)

df0 = df[(df['Data'] >= data_inicial) & (df['Data'] <= data_final)]
dfs = dfs[(dfs['Data'] >= data_inicial) & (dfs['Data'] <= data_final)]

##Gráficos

# Gráfico do Preço do Barril Brent
fig = go.Figure()
fig.update_layout(
    width=800,  # Largura em pixels
    height=500,  # Altura em pixels
)
# Plot 1: Preço do Petróleo Brent
fig.add_trace(go.Scatter(x=df0['Data'], y=df0['Brent (F0B)'], mode='lines', name='Preço do Petróleo', line=dict(color='midnightblue')))

# Plot 2: Índice de Xangai
if show_shanghai_index:
    fig.add_trace(go.Scatter(x=dfs['Data'], y=dfs['Último'], mode='markers', name='Índice de Xangai', marker=dict(color='green', size=3), yaxis='y2'))

# Formatação do layout
fig.update_layout(title='Preço do Petróleo', xaxis_title='Data', legend=dict(x=0, y=1.1))

# Se o checkbox for marcado, atualize as configurações do eixo y para o Índice de Xangai
if show_shanghai_index:
    fig.update_layout(yaxis=dict(title='Preço do Petróleo (US$/barril)', color='midnightblue'),
                      yaxis2=dict(title='Índice de Xangai', color='green', overlaying='y', side='right'))

fig.update_xaxes(showgrid=True, zeroline=True, zerolinewidth=2, zerolinecolor='black')
fig.update_yaxes(showgrid=True, zeroline=True, zerolinewidth=2, zerolinecolor='black')


##Visualização streamlit

st.title('ANÁLISE DE PREÇO DO PETRÓLEO')
aba1, aba2 = st.tabs(['Visão Geral', 'Previsão'])

with aba1:
    coluna1, coluna2, coluna3, coluna4, coluna5 = st.columns(5)

    with coluna1:
        st.metric('Máximo',formata_numero(df0['Brent (F0B)'].max(),''))
        st.plotly_chart(fig)
    with coluna2:
        st.metric('Mínimo',formata_numero(df0['Brent (F0B)'].min(),''))
    with coluna3:
        st.metric('Média',formata_numero(df0['Brent (F0B)'].mean(),''))
    with coluna4:
        st.metric('',' _')
        fig_consumo_fontes_energia = plotagem(dados)
        st.plotly_chart(fig_consumo_fontes_energia)

    if show_shanghai_index:
        long_text = '''A queda do preço do barril de petróleo em 2009 foi influenciada por uma série de fatores,
          incluindo a economia da China. A crise financeira global reduziu a demanda por petróleo, pois a oferta de 
          crédito diminuiu e as pessoas e empresas evitaram investir no que não consideravam essencial. Isso levou a 
          uma queda na demanda por petróleo em todo o mundo. Durante a primeira metade de 2008, a demanda por petróleo 
          aumentou substancialmente, especialmente por parte da China, que estava registrando uma alta taxa de crescimento 
          econômico. No entanto, com o início da crise financeira global na segunda metade do ano, a economia desacelerou, 
          levando a uma redução na demanda por petróleo. Além disso, a incerteza sobre o volume das reservas de petróleo
            que ainda existem para ser exploradas e a falta de dados confiáveis sobre quanto petróleo os países guardam 
            em suas reservas para o caso de emergências também influenciaram a queda dos preços. A Opep, preocupada com 
            a queda dos preços, anunciou um corte recorde na produção, de 2,2 milhões de barris, que passou a vigorar em 
            janeiro de 2009. No entanto, a percepção das pessoas sobre a oferta e a demanda de petróleo influencia mais os 
            preços do que esses dois fatores em si. Portanto, as flutuações na economia chinesa tiveram um impacto direto nos 
            preços globais do petróleo. fonte(https://www.bbc.com/portuguese/noticias/2009/01/090123_entenda_petroleo_tc2)'''
    else:
        long_text = '''Introdução  
A importância do petróleo no desenvolvimento econômico mundial é indiscutível, estendendo-se por mais de seis décadas como 
a principal fonte de energia do planeta. Sua versatilidade é crucial, fornecendo não apenas energia e luz, mas também uma 
ampla gama de produtos essenciais para o funcionamento da sociedade moderna, incluindo asfalto, gasolina, plásticos, 
querosene e óleo combustível.
Na cadeia produtiva global, o petróleo é um insumo de difícil substituição, desempenhando um papel central em setores-chave, 
como transporte, indústria automobilística, naval e química. Sem ele, a mobilidade de bens e pessoas seria severamente comprometida, 
impactando negativamente a economia global.
Países altamente dependentes do petróleo, como a Venezuela, enfrentam um risco significativo devido à sua exposição às flutuações dos
preços e disponibilidade deste recurso. De fato, no caso venezuelano, o petróleo representa cerca de 90% das divisas e quase metade do
orçamento governamental.
Além disso, a importância estratégica do petróleo tem sido uma fonte de tensões geopolíticas ao longo da história moderna, com conflitos
sendo frequentemente desencadeados em torno do controle e acesso a esse recurso valioso.
Apesar de sua importância inegável, a crescente consciência ambiental e a necessidade de reduzir as emissões de carbono estão impulsionando
a busca por alternativas energéticas mais sustentáveis. Portanto, enquanto o petróleo continua sendo fundamental para a economia global, 
é essencial investir em fontes de energia mais limpas e renováveis para garantir um futuro econômico e ambientalmente sustentável.

Análise das alterações de preços durante o período observado:  

Alta em julho de 2008  
O preço do petróleo vinha crescendo continuamente desde 2002. No entanto, entre o final de 2007 e a metade de 2008, esse crescimento 
se acelerou consideravelmente. Em julho de 2008, o preço do barril atingiu um recorde histórico, ultrapassando os US$ 150. Três fatores
principais contribuem para essa crise: a baixa elasticidade preço-demanda, o aumento da demanda por petróleo na China, no Oriente Médio
e em outros países industrializados, e a baixa capacidade ociosa, que dificulta um aumento significativo na produção.

Baixa em dezembro de 2008  
A recessão desencadeada pela crise financeira global de 2008, conhecida como a crise dos "subprime", foi a principal responsável pela 
queda abrupta no preço do petróleo durante esse período. O colapso da bolha imobiliária nos EUA deu início à crise, levando os Estados 
Unidos e a Europa a uma severa recessão, que por sua vez reduziu drasticamente a demanda por petróleo, fazendo seus preços despencarem.

Alta em maio de 2011  
Após a crise do “subprime” em 2008, várias economias globais começaram a se recuperar, aumentando a demanda por energia, especialmente 
nos países em desenvolvimento. Além disso, eventos geopolíticos, como instabilidades no Oriente Médio e conflitos em importantes regiões
produtoras de petróleo, geraram preocupações sobre a segurança do fornecimento, afetando os preços. A contínua desvalorização do dólar 
americano também desempenhou um papel significativo, tornando o petróleo mais caro para compradores que usavam outras moedas. Essa
combinação de fatores contribuiu para a retomada da alta nos preços dos combustíveis, culminando em maio de 2011.

Baixa em janeiro de 2016  
A queda nos preços do barril de julho de 2014 a janeiro de 2016 foi principalmente impulsionada por um excesso de oferta global, 
aliado a uma desaceleração da demanda. A produção de petróleo de xisto nos Estados Unidos aumentou de forma significativa, tornando 
o país menos dependente das importações e contribuindo para o aumento da oferta global. Paralelamente, a Organização dos Países Exportadores
de Petróleo (OPEP) manteve elevados níveis de produção, em parte para preservar sua participação de mercado diante do crescente volume 
de produção de xisto. No entanto, a desaceleração econômica global, especialmente na China, reduziu a demanda por petróleo. O excesso
de oferta, combinado com a falta de coordenação entre os principais produtores para reduzir a produção, resultou em uma queda acentuada 
nos preços, atingindo seu ponto mais baixo em janeiro de 2016.

Baixa em abril de 2020  
A queda brusca nos preços do barril de petróleo entre janeiro e março de 2020 foi fortemente influenciada por uma combinação de eventos
relacionados à pandemia de COVID-19 e uma disputa de preços entre a Rússia e a Arábia Saudita. A propagação global do coronavírus resultou
em medidas de confinamento e restrições de viagem, o que reduziu drasticamente a demanda por petróleo devido à paralisação de indústrias,
redução das viagens e forte impacto na atividade econômica. Em meio a esse cenário, a Arábia Saudita e a Rússia discordaram sobre os 
cortes na produção para sustentar os preços. Isso desencadeou uma guerra de interesses na qual ambos os países aumentaram sua produção, 
inundando ainda mais o mercado em um momento de queda acentuada na demanda.  
     
Alta em março de 2022  
À medida que as restrições da pandemia foram sendo gradualmente flexibilizadas, houve um aumento na demanda por energia, impulsionando 
assim a recuperação econômica. Além disso, as campanhas de vacinação bem-sucedidas em vários países têm contribuído para melhores perspectivas
econômicas globais, o que por sua vez fortaleceu a confiança dos investidores nos mercados de commodities. Ao mesmo tempo, importantes
produtores de petróleo, como os membros da OPEP e aliados como a Rússia, ajustaram sua produção por meio de cortes coordenados para 
equilibrar oferta e demanda. Essas medidas combinadas estão desempenhando um papel fundamental na sustentação e fortalecimento da 
recuperação econômica em curso.'''


    with st.container(height=300):
         st.markdown(long_text)

    #st.table(dados.head())
 
with aba2:

    wmape_value_prophet = wmape(forecast_prophet2['y_y'].values, forecast_prophet2['yhat'].values)
    formatted_wmape = f"{wmape_value_prophet:.3f}%"
    st.metric('Wmape:', formatted_wmape)
    st.plotly_chart(fig_prophet)
    
    wmape_value_naive = wmape(forecast_df['y'].values, forecast_df['Naive'].values)
    formatted_wmape = f"{wmape_value_naive:.3f}%"
    st.metric('Wmape:', formatted_wmape)
    st.plotly_chart(fig_naive)

#st.table(treino)
