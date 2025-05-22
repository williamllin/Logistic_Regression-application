# Logistic_Regression-application
# 1. Import Libraries
import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# 2. Download and Prepare Historical Data
tickers = {
    'USO': 'USO',
    'GDX': 'GM_ETF',
    'DX-Y.NYB': 'USD',
    'PA=F': 'Pd',
    'PL=F': 'Pt',
    'ILTB': 'USD Bond',
    'SIL=F': 'Sil',
    'CL=F': 'CO',
    'BZ=F': 'BCO',
    'EURUSD=X': 'EUR/USD',
    'EGO': 'Eldorado',
    '^DJI': 'DJ',
    '^GSPC': 'S&P 500',
    'GLD': 'SPDR',
    'IAU': 'iShares',
    'SGOL': 'SGOL',
    'PHYS': 'Sprott'
}

data_frames = []
for ticker, prefix in tickers.items():
    data = yf.download([ticker], period='10y')
    data.columns = [f'{prefix}_{col}' for col in data.columns]
    data_frames.append(data)

combined_df = pd.concat(data_frames, axis=1)
today_df = combined_df.tail(1).copy()
combined_df = combined_df.iloc[:-1]
yesterday_df = combined_df.tail(1).copy()

adj_close_df = combined_df[[col for col in combined_df.columns if col.endswith('_Adj Close')]]
low_df = combined_df[[col for col in combined_df.columns if col.endswith('_Low')]]
open_df = combined_df[[col for col in combined_df.columns if col.endswith('_Open')]]

# 3. Feature Engineering (X & Y)
low_df_shifted = low_df.shift(1).reindex(open_df.index)
increased_values = open_df.values - low_df_shifted.values
new_columns = [col.replace('_Open', '_increased') for col in open_df.columns]
increased_df = pd.DataFrame(increased_values, columns=new_columns, index=open_df.index)
X_df = increased_df.dropna().drop(columns=['iShares_increased', 'SGOL_increased', 'Sprott_increased'])

diff_df = adj_close_df.diff().dropna()
Y_df = pd.DataFrame([
    diff_df['iShares_Adj Close'],
    diff_df['SGOL_Adj Close'],
    diff_df['Sprott_Adj Close']
]).T
Y_df.columns = ["iShares_increased", "SGOL_increased", "Sprott_increased"]

# 4. Classification and Encoding
def classify_y(y):
    return "Up" if y > 0 else "Down"

y_iShares = Y_df.filter(like='iShares').applymap(classify_y)
y_SGOL = Y_df.filter(like='SGOL').applymap(classify_y)
y_Sprott = Y_df.filter(like='Sprott').applymap(classify_y)

X_train_df, X_test_df, y_iShares_train, y_iShares_test = train_test_split(X_df, y_iShares, test_size=0.2, random_state=77)
_, _, y_SGOL_train, y_SGOL_test = train_test_split(X_df, y_SGOL, test_size=0.2, random_state=77)
_, _, y_Sprott_train, y_Sprott_test = train_test_split(X_df, y_Sprott, test_size=0.2, random_state=77)

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_df), index=X_train_df.index, columns=X_train_df.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test_df), index=X_test_df.index, columns=X_test_df.columns)

def classify_x(x):
    if x > 3: return "Strong Up"
    elif x > 2: return "Up++"
    elif x > 1: return "Up+"
    elif x > 0: return "Up"
    elif x < -3: return "Strong Down"
    elif x < -2: return "Down--"
    elif x < -1: return "Down-"
    else: return "Down"

X_train_cat = X_train_scaled.applymap(classify_x)
X_test_cat = X_test_scaled.applymap(classify_x)

ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
ohe.fit(X_train_cat)

X_train_encoded = pd.DataFrame(ohe.transform(X_train_cat), index=X_train_cat.index, columns=ohe.get_feature_names_out())
X_test_encoded = pd.DataFrame(ohe.transform(X_test_cat), index=X_test_cat.index, columns=ohe.get_feature_names_out())

le = LabelEncoder()
y_iShares_train_enc = le.fit_transform(y_iShares_train.values.ravel())
y_iShares_test_enc = le.transform(y_iShares_test.values.ravel())
y_SGOL_train_enc = le.fit_transform(y_SGOL_train.values.ravel())
y_SGOL_test_enc = le.transform(y_SGOL_test.values.ravel())
y_Sprott_train_enc = le.fit_transform(y_Sprott_train.values.ravel())
y_Sprott_test_enc = le.transform(y_Sprott_test.values.ravel())

# 5. Model Training with Hyperparameter Tuning
param_grid = {
    'C': [0.1, 0.5, 1, 5, 10],
    'class_weight': [None, 'balanced'],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'lbfgs']
}

def train_and_predict(X_train, y_train, X_test):
    grid = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='accuracy')
    grid.fit(X_train, y_train)
    model = LogisticRegression(**grid.best_params_)
    model.fit(X_train, y_train)
    return model, model.predict(X_test), grid.best_params_

best_model, y_iShares_pred, best_params = train_and_predict(X_train_encoded, y_iShares_train_enc, X_test_encoded)
best_model_SGOL, y_SGOL_pred, best_params_SGOL = train_and_predict(X_train_encoded, y_SGOL_train_enc, X_test_encoded)
best_model_Sprott, y_Sprott_pred, best_params_Sprott = train_and_predict(X_train_encoded, y_Sprott_train_enc, X_test_encoded)

# 6. Make Prediction for the Latest Day
today_open = today_df.filter(like='_Open')
yesterday_low = yesterday_df.filter(like='_Low')
latest_result_values = today_open.values - yesterday_low.values
latest_X_df = pd.DataFrame(latest_result_values, columns=new_columns, index=today_df.index)
latest_X_df = latest_X_df.drop(columns=['iShares_increased', 'SGOL_increased', 'Sprott_increased'])

latest_scaled = pd.DataFrame(scaler.transform(latest_X_df), index=latest_X_df.index, columns=latest_X_df.columns)
latest_cat = latest_scaled.applymap(classify_x)
latest_encoded = pd.DataFrame(ohe.transform(latest_cat), index=latest_X_df.index, columns=ohe.get_feature_names_out())

latest_predictions = {
    "iShares": le.inverse_transform(best_model.predict(latest_encoded))[0],
    "SGOL": le.inverse_transform(best_model_SGOL.predict(latest_encoded))[0],
    "Sprott": le.inverse_transform(best_model_Sprott.predict(latest_encoded))[0]
}

print()
print("Predictions for the most recent day:")
print("iShares:", latest_predictions["iShares"])
print("SGOL:", latest_predictions["SGOL"])
print("Sprott:", latest_predictions["Sprott"])
print()
