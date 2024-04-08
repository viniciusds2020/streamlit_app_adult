import streamlit as st
import pandas as pd
import joblib

# Carregar o modelo treinado
model = joblib.load('random_forest_model.pkl')

# Função para preprocessar os dados de entrada
def preprocess_input(data):
    # Converter as colunas categóricas em one-hot encoding
    categorical_cols = ['age', 'workclass', 'education', 'marital-status', 'occupation',
     'relationship', 'race', 'sex', 'hours-per-week', 'native-country']
    data_processed = pd.get_dummies(data, columns=categorical_cols)
    return data_processed

# Função para fazer a predição
def predict_income(data):
    # Pré-processar os dados
    processed_data = preprocess_input(data)
    # Fazer a predição usando o modelo
    prediction = model.predict(processed_data)
    return prediction

# Configuração do aplicativo Streamlit
def main():
    st.title('Previsão de Renda')

    # Formulário para entrada de dados
    st.write('Insira os dados para prever a renda:')
    age = st.slider('Idade', 18, 90, 25)
    workclass = st.selectbox('Classe de Trabalho', ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'State-gov', 'Local-gov', 'Without-pay', 'Never-worked'])
    education = st.selectbox('Educação', ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'])
    marital_status = st.selectbox('Estado Civil', ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'])
    occupation = st.selectbox('Ocupação', ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'])
    relationship = st.selectbox('Relacionamento', ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'])
    race = st.selectbox('Raça', ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'])
    sex = st.selectbox('Gênero', ['Female', 'Male'])
    hours_per_week = st.slider('Horas por Semana', 1, 100, 40)
    native_country = st.selectbox('País de Origem', ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands'])

    # Criar um dataframe com os dados de entrada
    input_data = pd.DataFrame({
        'age': [age],
        'workclass': [workclass],
        'education': [education],
        'marital-status': [marital_status],
        'occupation': [occupation],
        'relationship': [relationship],
        'race': [race],
        'sex': [sex],
        'hours-per-week': [hours_per_week],
        'native-country': [native_country]
    })

    # Fazer a previsão ao clicar no botão
    if st.button('Prever'):
        input_data1 = preprocess_input(input_data)
        prediction = predict_income(input_data1)
        st.write('Resultado da Previsão:')
        if prediction[0] == 0:
            st.write('A renda estimada é inferior a $50,000 por ano.')
        else:
            st.write('A renda estimada é superior ou igual a $50,000 por ano.')

if __name__ == '__main__':
    main()
