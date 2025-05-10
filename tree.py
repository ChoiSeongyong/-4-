import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 데이터 로딩
df = pd.read_csv("labeled_customer_data.csv")

# 특성과 타깃 분리
X = df.drop(columns=["name", "email", "last_login", "days_since_login", "churned"])
y = df["churned"]

# 인코딩
cat_cols = X.select_dtypes(include='object').columns.tolist()
num_cols = X.select_dtypes(include='number').columns.tolist()

encoder = OneHotEncoder(sparse_output=False)
X_cat = encoder.fit_transform(X[cat_cols])

scaler = StandardScaler()
X_num = scaler.fit_transform(X[num_cols])

X_all = np.hstack((X_cat, X_num))

# 데이터 분할 (8:1:1)
X_temp, X_test, y_temp, y_test = train_test_split(X_all, y, test_size=0.1, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=1/9, random_state=42, stratify=y_temp)

# 모델 학습 (기본 설정: max_depth=5)
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)

# 성능 평가
train_acc = accuracy_score(y_train, model.predict(X_train))
val_acc = accuracy_score(y_val, model.predict(X_val))
test_acc = accuracy_score(y_test, model.predict(X_test))

print(f"Train Accuracy: {train_acc:.4f}")
print(f"Val Accuracy:   {val_acc:.4f}")
print(f"의사결정트리 점수:  {test_acc:.4f}")

# 혼동 행렬 시각화
cm = confusion_matrix(y_test, model.predict(X_test), labels=[0, 1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Churned", "Churned"])
disp.plot(cmap='Blues')
plt.title("혼동행렬 맵")
plt.show()
