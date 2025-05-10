import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import openai
import os
from datetime import datetime

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
print(f"Test Accuracy:  {test_acc:.4f}")

# 혼동 행렬 시각화
cm = confusion_matrix(y_test, model.predict(X_test), labels=[0, 1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Churned", "Churned"])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

"---------------------------------OPENAI API코드------------------------------------"

# 환경 변수에서 OPENAI API 키를 가져옴
openai.api_key = os.environ.get("OPENAI_API_KEY")

# 각 고객별 이탈 확률 예측
churn_probabilities = model.predict_proba(X_all)[:, 1]

# 이탈 확률을 df(read_csv)에 추가
df['churn_probability'] = churn_probabilities

# 고위험 고객 필터링 (확률은 기업 관리자가 입력한 값 가져오는 방식으로 바꿀 예정)
# 고객 중 확률이 0.7 이상인 고객의 데이터만 필터링 
high_risk_customers = df[df['churn_probability'] >= 0.7]

current_date = datetime.now()

# 필요한 데이터만 가져와서 gpt 메시지 생성
for index, customer in high_risk_customers.iterrows():
    name = customer['name']
    age = customer['age']
    preferred_category = customer['preferred_category']
    last_login = pd.to_datetime(customer['last_login'])
    email = customer['email']

    days_since_last_login = (current_date - last_login).days

    # GPT-3 모델을 이용해 맞춤형 메시지 생성
    prompt = f"""
    고객 데이터:
    - 고객 이름: {name}
    - 고객 나이: {age}세 (나이는 메시지에서 언급하면 안됨)
    - 고객 선호 카테고리: {preferred_category}
    - 고객의 마지막 로그인 후 {days_since_last_login}일이 지남.

    고객 데이터 기반 고객님에게 다음과 같은 내용을 포함하는 메시지 생성:
    1. 고객의 이름, 마지막 로그인 후 지난일 수를 언급하면서 걱정한 듯한 메시지
    2. 고객의 나이와 선호 카테고리 기반 고객 맞춤형 추천 리스트 3가지 추천 메시지
    3. 고객으로부터 서비스 이용을 유도할 수 있는 메시지와 지금 시정하러가기(넷플릭스 홈피로 이동) 클릭할 수 있는 메시지
    """

    # GPT-3 모델로 메시지 생성
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "고객 이탈을 막기 위한 맞춤형 메시지 생성."},
            {"role": "user", "content": prompt}
        ]
    )

    churn_message = response['choices'][0]['message']['content'].strip()
    print(f"이메일: {email}\n생성된 메시지:\n{churn_message}\n")

# 추후 이메일 연동 코드 추가
