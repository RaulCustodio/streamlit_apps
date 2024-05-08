import  streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose


st.title('DASHBOARD DE PREVISÃO DE PREÇO DO PETRÓLEO')
