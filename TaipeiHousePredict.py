# 4/15# 匯入需要的套件
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 1. 讀取資料
df = pd.read_csv("Taipei_house.csv")

# 🌟 加入一個新的特色欄位：模擬樓層（假設數據中沒有，我們隨機產生）
import numpy as np
np.random.seed(42)
df["樓層"] = np.random.randint(1, 21, size=len(df))  # 模擬 1~20 樓之間

# 2. 資料前處理
df = df.dropna()  # 移除缺失值
df = pd.get_dummies(df, columns=["區域"])  # 區域做 one-hot encoding

# 3. 建立特徵與標籤
X = df.drop("單價", axis=1)  # 特徵（去掉"單價"欄）
y = df["單價"]               # 標籤（預測的目標）

# 4. 分割訓練與測試資料
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. 建立與訓練模型
model = LinearRegression()
model.fit(X_train, y_train)

# 6. 預測與模型評估
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"平均誤差 MSE: {mse:.2f}")

# 7. 畫出預測結果與實際值的比較圖
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.6, color='blue', label='預測點')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='理想線')
plt.xlabel("實際單價（萬元/坪）")
plt.ylabel("預測單價（萬元/坪）")
plt.title("台北市房價預測比較圖")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
