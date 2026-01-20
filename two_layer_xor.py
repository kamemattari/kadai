import numpy as np

# シグモイド関数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# シグモイドの微分
def sigmoid_deriv(x):
    return x * (1 - x)

# XORデータ
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

# 重みの初期化
np.random.seed(0)
W_hidden = np.random.randn(2, 2)
b_hidden = np.zeros((1, 2))
W_output = np.random.randn(2, 1)
b_output = np.zeros((1, 1))

lr = 0.1

# 学習
for epoch in range(1000):
    # 順伝播
    hidden_output = sigmoid(np.dot(X, W_hidden) + b_hidden)
    y_hat = sigmoid(np.dot(hidden_output, W_output) + b_output)


    # 誤差
    error = y - y_hat

    # 誤差逆伝播
    d_output = error * sigmoid_deriv(y_hat)
    d_hidden = np.dot(d_output, W_output.T) * sigmoid_deriv(hidden_output)

    # 重み更新
    W_output += lr * np.dot(hidden_output.T, d_output)
    b_output += lr * np.sum(d_output, axis=0, keepdims=True)    
    W_hidden += lr * np.dot(X.T, d_hidden)
    b_hidden += lr * np.sum(d_hidden, axis=0, keepdims=True)

# 結果表示
print("Predictions:")
print(y_hat.round(3))

