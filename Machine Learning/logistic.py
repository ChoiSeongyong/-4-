import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

# 데이터 불러오기
df = pd.read_csv("labeled_customer_data.csv")

# 특성과 라벨 분리
X = df.drop(columns=["name", "email", "last_login", "days_since_login", "churned"])
y = df["churned"]

# 범주형, 수치형 분리
cat_cols = X.select_dtypes(include='object').columns.tolist()
num_cols = X.select_dtypes(include='number').columns.tolist()

# 인코딩
encoder = OneHotEncoder(sparse_output=False)
X_cat = encoder.fit_transform(X[cat_cols])

scaler = StandardScaler()
X_num = scaler.fit_transform(X[num_cols])

# 최종 feature matrix
X_all = np.hstack((X_cat, X_num))

# 8:1:1 split
X_temp, X_test, y_temp, y_test = train_test_split(X_all, y, test_size=0.1, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=1/9, random_state=42, stratify=y_temp)

# 모델 생성
# eta0=0.1 또는 0.001 등으로 실험
model = SGDClassifier(loss='log_loss', max_iter=1, eta0=0.1, learning_rate='constant', random_state=42, warm_start=True)

# 에포크 학습
epochs = 20
for epoch in range(1, epochs + 1):
    model.partial_fit(X_train, y_train, classes=np.array([0, 1]))

# 테스트 평가
test_acc = accuracy_score(y_test, model.predict(X_test))
print(f"\n 로지스틱 리그레션 정확도: {test_acc:.4f}")
