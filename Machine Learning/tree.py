import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

# 설정
MAX_DEPTH = 5

# 데이터 로딩
df = pd.read_csv("processed_customer_data.csv")

# 특성과 타깃 분리
X = df.drop(columns=["churned"])
y = df["churned"]

# 데이터 분할 (8:1:1)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=1/9, stratify=y_temp, random_state=42)

# 모델 학습 (기본 설정: max_depth=5)
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)

# 성능 평가
y_pred = model.predict(X_test)
train_acc = accuracy_score(y_train, model.predict(X_train))
val_acc = accuracy_score(y_val, model.predict(X_val))
test_acc = accuracy_score(y_test, model.predict(X_test))

print(f"Train Accuracy: {train_acc:.4f}")
print(f"Val Accuracy:   {val_acc:.4f}")
print(f"Test Accuracy(의사결정트리 점수):  {test_acc:.4f}")

# 리포트 출력
print("\n 분류 리포트:")
print(classification_report(y_test, y_pred, target_names=["Not Churned", "Churned"]))

# 혼동 행렬 시각화
cm = confusion_matrix(y_test, model.predict(X_test), labels=[0, 1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Churned", "Churned"])
disp.plot(cmap='Blues')
plt.title("혼동행렬 맵")
plt.show()
