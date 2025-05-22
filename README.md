# Logistic_Regression-application
# 1. Download and Merge Financial Data
import pandas as pd
import yfinance as yf

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
combined_df = combined_df.iloc[:-1]

# 2. Extract Key Price Columns and Compute Features
adj_close_df = combined_df[[col for col in combined_df.columns if col.endswith('_Adj Close')]]
low_df = combined_df[[col for col in combined_df.columns if col.endswith('_Low')]]
open_df = combined_df[[col for col in combined_df.columns if col.endswith('_Open')]]

low_df_shifted = low_df.shift(1).reindex(open_df.index)
increased_values = open_df.values - low_df_shifted.values
increased_columns = [col.replace('_Open', '_increased') for col in open_df.columns]
increased_df = pd.DataFrame(increased_values, columns=increased_columns, index=open_df.index)
X_df = increased_df.dropna().drop(columns=['iShares_increased', 'SGOL_increased', 'Sprott_increased'])

# 3. Compute Target Labels (Y)
diff_df = adj_close_df.diff()
Y_df = pd.DataFrame([
    diff_df['iShares_Adj Close'],
    diff_df['SGOL_Adj Close'],
    diff_df['Sprott_Adj Close']
]).T.dropna()
Y_df.columns = ["iShares_increased", "SGOL_increased", "Sprott_increased"]

# 4. Merge and Classify Data for Association Rules
list_df = pd.concat([X_df, Y_df], axis=1)

def classify_value(y):
    return 1 if y > 0 else 0

classified_df = list_df.applymap(classify_value)

# 5. Association Rule Mining and Filtering
from mlxtend.frequent_patterns import apriori, association_rules

frequent_itemset = apriori(classified_df, min_support=0.3, use_colnames=True)
rules = association_rules(frequent_itemset, metric="lift", min_threshold=1.6)
valuable_rules = rules[rules['confidence'] >= 0.6]

def filter_rules(rules):
    rules = rules[~rules['antecedents'].astype(str).str.contains('iShares_increased|SGOL_increased|Sprott_increased')]
    rules = rules[rules['consequents'].astype(str).str.contains('iShares_increased|SGOL_increased|Sprott_increased')]
    return rules

filtered_rules = filter_rules(valuable_rules)
print(filtered_rules.sort_values(by='lift', ascending=False).head(10))
