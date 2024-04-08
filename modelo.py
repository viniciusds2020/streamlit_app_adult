import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Carregar o dataset Adult
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 
           'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
data = pd.read_csv(url, names=columns, sep=',\s', na_values=["?"], engine='python')

# Remover linhas com valores nulos
data = data.dropna()

# Selecionar colunas para features (características) e target (alvo)
features = ['age', 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 
            'race', 'sex', 'hours-per-week', 'native-country']
target = 'income'

# Converter coluna alvo (income) em binária (0 ou 1)
label_enc = LabelEncoder()
data['income'] = label_enc.fit_transform(data['income'])

# Converter colunas categóricas em numéricas usando one-hot encoding
data_processed = pd.get_dummies(data[features])

# Dividir os dados em conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(data_processed, data[target], test_size=0.2, random_state=42)

# Inicializar e treinar o modelo Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=210)
rf_model.fit(X_train, y_train)

# Avaliar o modelo
train_accuracy = rf_model.score(X_train, y_train)
test_accuracy = rf_model.score(X_test, y_test)
print(f'Acurácia do modelo no conjunto de treino: {train_accuracy:.2f}')
print(f'Acurácia do modelo no conjunto de teste: {test_accuracy:.2f}')

# Salvar o modelo treinado em um arquivo .pkl
joblib.dump(rf_model, 'random_forest_model.pkl')
print("Modelo salvo com sucesso como 'random_forest_model.pkl'")
