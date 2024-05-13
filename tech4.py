import  streamlit as st
import pandas as pd
import plotly.graph_objs as go
from st_pages import Page, Section, show_pages, add_page_title

tab1, tab2, tab3 = st.tabs(["Dashboard", "Prediction", "Data"])

with tab1:
    st.title('PREÇO DO PETRÓLEO')

    #Uploading df
    df = pd.read_csv("brentdb.csv", sep=';')
    df = df.rename(columns={'Preço - petróleo bruto - Brent (FOB)': 'Brent (F0B)'})
    df['Data'] = pd.to_datetime(df['Data'], origin='1899-12-30', unit='D')
    df['Brent (F0B)'] = df['Brent (F0B)'].str.replace(',', '.')
    df['Brent (F0B)'] = df['Brent (F0B)'].str.replace(',', '.').astype(float)


    st.sidebar.title('Filters')
    start_date = pd.to_datetime(st.sidebar.date_input(label = 'Data de início', value= df['Data'].min(), min_value = df['Data'].min(), max_value = df['Data'].max()))
    end_date = pd.to_datetime(st.sidebar.date_input('Data final', value= df['Data'].max(), min_value = df['Data'].min(), max_value = df['Data'].max()))
    0
    df = df[(df['Data'] >= start_date) & (df['Data'] <= end_date)]




    max_value_ever = df['Brent (F0B)'].max()
    minimum_value_ever = df['Brent (F0B)'].min()
    filtered_month_names = df[df['Brent (F0B)'] == max_value_ever]['Data'].dt.month_name().unique()
    filtered_month_names_min = df[df['Brent (F0B)'] == minimum_value_ever]['Data'].dt.month_name().unique()
    average_price = df['Brent (F0B)'].mean().round()

    col1, col2, col3 = st.columns(3)
    col1.metric("Maximum Price", max_value_ever)
    col2.metric("Minimum Price", minimum_value_ever)
    col3.metric("Average Price", average_price)



    #ploting-linear
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['Data'],
        y=df['Brent (F0B)'],
        mode='lines',
        line=dict(color='midnightblue'),
        error_y=dict(
            type='constant',
            value=1,  
            thickness=1,
            width=2,
            color='orange'
        )
    ))


    fig.update_layout(
        title='Historico do Preço do Petróleo',
        xaxis_title='Data',
        yaxis_title='Preço - petróleo bruto - Brent (FOB)',
        title_font_size=14,
        xaxis=dict(
            tickformat='%Y-%m-%d' 
        )
    )


    st.plotly_chart(fig)


with tab2:
    st.write('the model comes here')




with tab3:
    st.title('Data')
    st.dataframe(df)
