import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
# Configuração inicial da página (deve ser a primeira chamada ao Streamlit)
st.set_page_config(
    layout="centered",
    initial_sidebar_state="expanded"
)

#função para carregar o dateset
@st.cache
def get_data():
    return pd.read_csv('data.csv')

#função para treinar o modelo
def train_model():
    data = get_data()
    x = data.drop('MEDV', axis=1)
    y = data['MEDV']
    rf_regressor = RandomForestRegressor()
    rf_regressor.fit(x,y)
    return rf_regressor

data = get_data()

model = train_model()

# Escolha de tema (Light ou Dark)
tema = st.sidebar.radio("Escolha o tema", ("Light", "Dark"))

# Aplicar cor ao texto com base no tema
if tema == "Dark":
    text_color = "white"
    bg_color = "#333333"
else:
    text_color = "black"
    bg_color = "#ffffff"

# Aplicar estilo básico ao título e subtítulo
st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {bg_color};
        color: {text_color};
    }}
    </style>
    """,
    unsafe_allow_html=True
)

#Definir o Título
st.title('Data App: Prevendo valores de imóveis')

#Subtitulo
st.markdown("Este é um Data App utilizando para exibir a solução de Machine learning para o problema de predição de valores de imóveis de Boston")

#Verificando o dataset
st.subheader("Selecionando apenas um pequeno conjunto de atributos")

#Atributos para serem exibidos por padrão
defaultcols = ["RM","PTRATIO","CHAS","MEDV"]

#definindo atributos a partir do multiselect
cols = st.multiselect("Atributos", data.columns.tolist(), default=defaultcols)

#eXIBINDO OS TOP 10 REGISTRO DO DATAFRAME
st.dataframe(data[cols].head(10))

st.subheader("Distribuição de imóveis por preço")

#Definindo a faixa de valores
faixa_valores = st.slider("Faixa de preço", float(data.MEDV.min()), 150., (10.0,100.0))

#Filtrando os dados
dados = data[data['MEDV'].between(left=faixa_valores[0], right=faixa_valores[1])]

#Plot a distribuição de dados
f = px.histogram(dados, x="MEDV", nbins=50, title='Distribuição de preços')
f.update_xaxes(title='MEDV')
f.update_yaxes(title='Total imóveis')
st.plotly_chart(f)
st.sidebar.subheader("Defina os atributos do imóvel para predição")

#Mapeando dados do usuário para cada atributo
crim = st.sidebar.number_input("Taxa de Criminalidade", value=data.CRIM.mean())
indus = st.sidebar.number_input("Propoção de Hectares de Negócios", value=data.INDUS.mean())
chas = st.sidebar.selectbox("Faz limite com o rio?", ("Sim", "Não"))

#Transformando o dado de entrada em valor binário
chas = 1 if chas == "Sim" else 0

nox = st.sidebar.number_input("concentração de óxido nítrico", value=data.NOX.mean())

rm = st.sidebar.number_input("Número de quartos", value=1)

ptratio = st.sidebar.number_input("Índice de alunos para professores", value=data.PTRATIO.mean())

#Inserindo m botão na tela
btn_predict = st.sidebar.button("Realizar Predição")

#Verificar se o botão foi acionado
if btn_predict:
    result = model.predict([[crim, indus, chas, nox, rm, ptratio]])
    st.subheader("O valor previsto para o imóvel é:")
    result = "US $" + str(round(result[0]* 10,2))
    st.write(result)

#!streamlit run /content/iris-ml-app.py