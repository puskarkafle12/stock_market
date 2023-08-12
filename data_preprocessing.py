
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def compute_rsi(data, window):
    diff = data.diff()
    up_chg = 0 * diff
    down_chg = 0 * diff
    up_chg[diff > 0] = diff[ diff>0 ]
    down_chg[diff < 0] = diff[ diff < 0 ]
    up_chg_avg   = up_chg.rolling(window=window).mean()
    down_chg_avg = down_chg.rolling(window=window).mean()
    rs = abs(up_chg_avg/down_chg_avg)
    rsi = 100 - 100/(1+rs)
    return rsi


def preprocess_data(df, window_size=5):
    df['Moving_Avg'] = df['Close'].rolling(window=window_size).mean()
    df['RSI'] = compute_rsi(df['Close'], window_size)
    df = df.dropna()

    features = df[['Open', 'Moving_Avg', 'RSI']]
    target = df['Close']

    # Scale the features
    scaler_features = StandardScaler()
    features_scaled = scaler_features.fit_transform(features)

    # Scale the target
    scaler_target = StandardScaler()
    target_scaled = scaler_target.fit_transform(target.values.reshape(-1, 1))

    # Create data structure with timestamps
    x_data = []
    for i in range(window_size, len(features_scaled)):
        x_data.append(features_scaled[i-window_size:i])
    x_data = np.array(x_data)

    return x_data, target, scaler_target
