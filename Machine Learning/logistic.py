import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report

# 데이터 불러오기
df = pd.read_csv("processed_customer_data.csv")

# 특성과 라벨 분리
X = df.drop(columns=["churned"])
y = df["churned"]

# 8:1:1 split
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=1/9, stratify=y_temp, random_state=42)

# 모델 생성
# eta0=0.1 또는 0.001 등으로 실험
model = SGDClassifier(loss='log_loss', max_iter=1, eta0=0.1, learning_rate='constant', random_state=42, warm_start=True)

# 에포크 학습
epochs = 20
for epoch in range(1, epochs + 1):
    model.partial_fit(X_train, y_train, classes=np.array([0, 1]))

# 테스트 평가
y_pred = model.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
print(f"\n 로지스틱 리그레션 정확도: {test_acc:.4f}")

# 리포트 출력
print("\n 분류 리포트:")
print(classification_report(y_test, y_pred, target_names=["Not Churned", "Churned"]))
