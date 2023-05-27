from math import sqrt

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import stats
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


# from yellowbrick.classifier import ClassificationReport


# preprocessamento-------------------------------------------------
# data cleaning (outliner removing)
def data_cleaning(df_t):
    df_t = df_t[df_t['ESLE'] > 0.1]
    df_t = df_t.reset_index(drop=True)
    df_t['ESLE_Zscore'] = stats.zscore(df_t['ESLE'])
    df_t = df_t[df_t['ESLE_Zscore'].between(-3, 3)]
    df_t = df_t.drop(columns=['ESLE_Zscore'])
    # print(df_t)
    return df_t


# data normalization
def data_normalization(dt_t):
    # selecionar a coluna com o target das restantes
    cols = [col for col in dt_t.columns if col not in ['ESLE']]
    d = dt_t[cols]  # data
    t = dt_t['ESLE']  # target

    # Scale the features to 0-1 range
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Apply normalization to all columns
    d = pd.DataFrame(scaler.fit_transform(d), columns=d.columns)
    # print(d)

    # --------------------------------------------------------------
    d.to_csv('teste.csv', decimal=',', sep=';', index=True)
    return d, t


# efetuar a divisao
def training_validation_test(d, t):
    data_train, data_test, target_train, target_test = train_test_split(d, t, test_size=0.30, random_state=42)
    return data_train, target_train, data_test, target_test


def to_graph(mlp1, mlp2, data_test, target_test):
    print(mlp1.score(data_test, target_test))  # score: Maior = Melhor
    print(mlp2.score(data_test, target_test))
    plt.hist([abs(target_test - mlp1.predict(data_test)), abs(target_test - mlp2.predict(data_test))], bins=20,
             label=['MLP 1', 'MLP 2'])
    plt.legend()
    plt.xlabel('Residue')
    plt.ylabel('Frequency')
    plt.show()


def validation(data_train, target_train, data_test, target_test):
    # Grid Search Cross Validation
    param_grid = {'activation': ['logistic'], 'alpha': [0.001, 0.0001, 0.00001],
                  'early_stopping': [False, True], 'hidden_layer_sizes': [(20,), (25,), (30,)],
                  'learning_rate': ['constant'], 'learning_rate_init': [0.001],
                  'max_iter': [9999, 20000], 'momentum': [0.9], 'n_iter_no_change': [10], 'solver': ['lbfgs']}

    model = MLPRegressor(random_state=42)

    grid_search = GridSearchCV(model, param_grid=param_grid, cv=5, scoring="r2")
    grid_search.fit(data_train, target_train)

    best_params = grid_search.best_params_
    model.set_params(**best_params)
    model.fit(data_train, target_train)

    print("Best param:", grid_search.best_params_)

    # -----------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------

    classifier = MLPRegressor(hidden_layer_sizes=20, activation='tanh', solver='lbfgs', max_iter=9999,
                              random_state=42)
    classifier.fit(data_train, target_train)

    # Cross Validation ------------------------------------------------------------------------------
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
    rmse = sqrt(mse)
    print("Root Mean Squared Error (RMSE):", rmse)

    # Compute Mean Absolute Error (MAE)
    mae = mean_absolute_error(target_test, target_pred)
    print("Mean Absolute Error (MAE):", mae)
    # -------------------------------------------------------

    # mariquices --------------------------------------------
    sns.set(font='Bauhaus 93')
    plt.style.use("dark_background")
    plt.rcParams['axes.facecolor'] = '#202020'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.color'] = 'white'
    plt.rcParams['grid.alpha'] = 0.05
    plt.rcParams['axes.edgecolor'] = '#909090'
    # ----------------------------------------------------

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=target_test, y=target_pred)
    plt.xlabel('Actual ESLE', fontsize=20)
    plt.ylabel('Predicted ESLE', fontsize=20)
    plt.title('Actual vs Predicted ESLE', fontsize=30)
    plt.show()

    error = target_test - target_pred
    sns.displot(error, bins=25)
    plt.title('Histogram of Prediction Errors')
    plt.show()

    to_graph(classifier, model, data_test, target_test)

    return classifier


def test(data_test, target_test):
    clf_final.predict(data_test)


def TestMe(file):
    df_tm = pd.read_csv(file)
    df_tm = data_cleaning(df_tm)
    data_tm, target_tm = data_normalization(df_tm)
    df_result = clf_final.predict(data_tm)
    rmse = sqrt(mean_squared_error(target_tm, df_result))
    return df_result, rmse


if __name__ == '__main__':
    df = pd.read_csv('Lab6-Proj1_Dataset.csv', delimiter=',')
    df = data_cleaning(df)
    data, target = data_normalization(df)
    d_train, t_train, d_test, t_test = training_validation_test(data, target)

    clf_final = validation(d_train, t_train, d_test, t_test)

    test(d_test, t_test)
    # TestMe("Lab6-Proj1_Testset.csv")
