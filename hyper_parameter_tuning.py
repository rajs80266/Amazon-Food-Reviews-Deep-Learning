import pandas as pd
import numpy as np

import csv
from sklearn.model_selection import GridSearchCV

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

from keras.layers import Dense, Input
from sklearn.metrics import mean_absolute_error

with open('./Reviews.csv', newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    data = [row for row in reader]

df = pd.DataFrame(data)

for column in ['HelpfulnessNumerator', 'HelpfulnessDenominator', 'Score']:
  df[column] = pd.to_numeric(df[column])

# numerical_features = ['HelpfulnessNumerator', 'HelpfulnessDenominator']
text_feature = 'Summary'
# categorical_features = ['ProductId', 'UserId']

# Clean the text data
df[text_feature] = df[text_feature].fillna('').astype(str)

# Tokenize the text feature
max_words = 10000
max_len = 100
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(df[text_feature].values)

# Convert the text feature to sequences
sequences = tok.texts_to_sequences(df[text_feature].values)
sequences = pad_sequences(sequences, maxlen=max_len)

# One-hot encode the categorical features
# encoder = OneHotEncoder(handle_unknown='ignore')
# X_cat = encoder.fit_transform(df[categorical_features])

# Scale the numerical features
# scaler = StandardScaler()
# X_num = scaler.fit_transform(df[numerical_features])

# Concatenate the encoded categorical features and the numerical features
X_train = sequences

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_train, df['Score'].values, test_size=0.2, random_state=42)


def create_model(optimizer):
    input_layer = Input(shape=(X_train.shape[1],))
    x = Dense(256, activation='relu')(input_layer)
    x = Dense(128, activation='relu')(input_layer)
    x = Dense(64, activation='relu')(input_layer)
    x = Dense(32, activation='relu')(input_layer)
    x = Dense(16, activation='relu')(x)
    output_layer = Dense(1, activation='linear')(x)
    model = Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    model.compile(loss='mean_absolute_error', optimizer=optimizer)
    return model

regressor = KerasRegressor(build_fn=create_model)

param_grid = {
    'optimizer': ['adam'],
    'batch_size': [64],
    'epochs': [50]
}

grid = GridSearchCV(estimator=regressor, param_grid=param_grid, cv=3)
grid_result = grid.fit(X_train, y_train)

print("Best parameters: ", grid_result.best_params_)
print("Best accuracy: ", grid_result.best_score_)

# Evaluate the model
# score = grid_result.evaluate(X_test, y_test, verbose=0)
# print("Test loss:", score)

y_pred = grid_result.predict(X_train)
rmse = np.sqrt(mean_absolute_error(y_train, y_pred))
print("Mean absolute error:", rmse)