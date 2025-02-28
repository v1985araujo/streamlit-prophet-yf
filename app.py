from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import pandas as pd, yfinance as yf, streamlit as st
from datetime import date

# Dicionário com os tickers e seus respectivos nomes
tickers = {"BTC-USD": "Bitcoin",
           "ETH-USD": "Ethereum",
           "SOL-USD": "Solana",  
           "PETR4.SA": "Petrobras",
           "TAEE11.SA": "Taesa",
           "BBAS3.SA": "Banco do Brasil",
           "ABEV3.SA": "Ambev",
           "MGLU3.SA": "Magasine Luísa",
           "AAPL": "Apple",
           "AMZN": "Amazon",
           "GOOG": "Google",
           "META": "Meta",
           "MSFT": "Microsoft",
           "NVDA": "Nvidia",
           "TSLA": "Tesla",
           "BTC": "ETF Bitcoin Grayscale"                
}

def load_data(ticker:str, inicial:date, final:date) -> pd.DataFrame:
    """ Carregar dados do Yahoo Finance
        :param ticker: str: Ticker da ação
        :param inicial: date: Data inicial
        :param final: date: Data final
        :return: pd.DataFrame: Dados da ação para o período selecionado
    """
    df = yf.Ticker(ticker).history(start = inicial.strftime('%Y-%m-%d'), 
                                   end = final.strftime('%Y-%m-%d'))
    return df


def forecast(df:pd.DataFrame, meses:int) -> tuple:
    """ Realizar previsão de valores futuros com o Prophet
        :param df: pd.DataFrame: Dados da ação
        :param meses: int: Meses para previsão
        :return: tuple: Modelo e previsão
    """
    df.reset_index(inplace = True)
    df = df.loc[:, ['Date', 'Close']]# Filtrando o dataframe
    df['Date'] = df['Date'].dt.tz_localize(None) # Remover Timezone das datas
    df.rename(columns={'Date':'ds', 'Close':'y'}, inplace = True) # Formato que o Prophet aceita
    modelo = Prophet()
    modelo.fit(df)
    future = modelo.make_future_dataframe(periods = int(meses) * 30)
    prediction = modelo.predict(future)
    return modelo, prediction


st.markdown("""
#   Aplicação de Análise Preditiva
### Prevendo valor de ações na bolsa de valores
""")
st.image('logo.jpg')
with st.sidebar:
    st.header("Dados para a Pesquisa")
    ticker = st.selectbox("Seleciona o ativo", tickers.keys()) # As chaves do dicionário alimentam o selectbox
    dt_inicial = st.date_input("Data inicial do histórico", value = date(2004, 1, 1))
    dt_final = st.date_input("Data final", max_value = date.today())
    meses = st.number_input("Meses de Previsão", 1, 24, value = 12)


dados = load_data(ticker, dt_inicial, dt_final)
if dados.shape[0] != 0:
    st.header(f"Dados de {tickers[ticker]}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = dados.index, y = dados['Close']))
    st.plotly_chart(fig)
    if meses == 1:
        string = "Previsão para o próximo mês "
    else:
        string = f"Previsão para os próximos {meses} meses "
    
    st.header(f"{string}")
    model, clairvoyance = forecast(dados, meses)
    fig = plot_plotly(model, clairvoyance, xlabel = 'Período', ylabel = 'Valor')
    st.plotly_chart(fig)
else:
    st.warning("Nenhum dado foi encontrado!:poop:")