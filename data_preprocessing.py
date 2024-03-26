import pandas as pd
import pandas_ta as ta

symbols = ['AAPL', 'AXP', 'BA', 'CAT', 'CSCO', 'CVX', 'DIS', 'DD', 'GS', 'HD',
           'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 'MRK', 'MSFT', 'NKE',
           'PG', 'TRV', 'UNH', 'RTX', 'VZ', 'WBA', 'WMT', 'XOM', 'GE', 'PFE']

for symbol in symbols:
    filename = f'stockdata/{symbol}_stock_data.csv'
    data = pd.read_csv(filename, skipfooter=1, engine='python')
    data['adjClose'].fillna(method='ffill', inplace=True)
    data['adjClose'] = pd.to_numeric(data['adjClose'], errors='coerce')

    data['daily_return'] = data['adjClose'].pct_change() * 100
    data['daily_return'].fillna(0, inplace=True)
    data['ema'] = ta.ema(data['adjClose'], length=3)
    data['rsi'] = ta.rsi(data['adjClose'], length=3)


    data = data.iloc[:-1]
    output_filename = f'preprocessed_data/{symbol}_preprocessed_data.csv'
    data.to_csv(output_filename, index=False)
