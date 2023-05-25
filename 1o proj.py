import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import stats
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

"""
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from yellowbrick.classifier import ClassificationReport
"""

df = pd.read_csv('Lab6-Proj1_Dataset.csv', delimiter=',')

# preprocessamento-------------------------------------------------

# data cleaning (outliner removing)
df = df[df['ESLE'] > 0.1]
df['ESLE_Zscore'] = stats.zscore(df['ESLE'])
df = df[df['ESLE_Zscore'].between(-3, 3)]
df = df.drop(columns=['ESLE_Zscore'])
df = df.reset_index(drop=True)
print(df)

# data normalization

# selecionar a coluna com o target das restantes
cols = [col for col in df.columns if col not in ['ESLE']]
data = df[cols]
target = df['ESLE']

"""
# Scale the features to 0-1 range
scaler = MinMaxScaler(feature_range=(0, 1))

# Apply normalization to all columns
data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
"""
# --------------------------------------------------------------

# data.to_csv('teste.csv', decimal=',', sep=';', index=True)

# efetuar a divisao
data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.70, random_state=42)

classifier = MLPRegressor(hidden_layer_sizes=(150), activation='logistic', solver='sgd', max_iter=9999, random_state=42)
classifier.fit(data_train, target_train)


# Cross Validation ---------------------------------------------------
scores = cross_val_score(classifier, data_train, target_train, cv=5)

# Print the accuracy scores for each fold
print("Cross-validation scores: ", scores)

# Calculate and print the mean accuracy score
mean_score = scores.mean()
print("Mean accuracy: ", mean_score)
# ----------------------------------------------------------------


# Testing -------------------------------------------------
target_pred = classifier.predict(data_test)
# Compute Mean Squared Error (MSE)
mse = mean_squared_error(target_test, target_pred)
print("Mean Squared Error (MSE):", mse)

# Compute Mean Absolute Error (MAE)
mae = mean_absolute_error(target_test, target_pred)
print("Mean Absolute Error (MAE):", mae)
# -------------------------------------------------------



plt.figure(figsize=(10, 6))
sns.scatterplot(x=target_test, y=target_pred)
plt.xlabel('Actual ESLE')
plt.ylabel('Predicted ESLE')
plt.title('Actual vs Predicted ESLE')
plt.show()


error = target_test - target_pred
sns.displot(error, bins=25)
plt.title('Histogram of Prediction Errors')
plt.show()
