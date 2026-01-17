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
for epoch in range(10000):
    # 順伝播
    hidden_output = sigmoid(np.dot(X, W_hidden) + b_hidden)
    y_hat = sigmoid(np.dot(hidden_output, W_output) + b_output)


    # 誤差
    error = y - y_hat

    # 逆伝播
    d2 = error * sigmoid_deriv(y_hat)
    d1 = np.dot(d2, W2.T) * sigmoid_deriv(h)

    # 更新
    W2 += lr * np.dot(h.T, d2)
    b2 += lr * np.sum(d2, axis=0, keepdims=True)
    W1 += lr * np.dot(X.T, d1)
    b1 += lr * np.sum(d1, axis=0, keepdims=True)

# 結果表示
print("Predictions:")
print(y_hat.round(3))

