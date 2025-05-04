import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("dummy_customer_data.csv", encoding='utf-8-sig')
#encoding='utf-8-sig': 엑셀에서 한글 안깨지도록 설정

# 최근 로그인일 → 며칠 전
df['last_login'] = pd.to_datetime(df['last_login'])
df['days_since_login'] = (datetime.today() - df['last_login']).dt.days

# 식별자 정보 따로 저장 (gpt메시지 생성용 -> 이후 머신러닝 결과와 함께 사용)
identity_df = df[['name', 'email', 'age', 'preferred_category']].copy()
identity_df.to_csv("customer_identity.csv", index=False, encoding='utf-8-sig')

# 범주형 → 수치형 인코딩
df = pd.get_dummies(df, columns=['preferred_category']) #카테고리 수치화
df['payment_status'] = df['payment_status'].map({'paid': 0, 'unpaid': 1}) #payment상태 수치화

# 불필요한 열 제거
df = df.drop(columns=['name', 'email', 'last_login'])

# 값들을 0~1 사이로 정규화 (로지스틱회귀시 필요)
scaler = MinMaxScaler()
df[['age', 'watch_time', 'days_since_login']] = scaler.fit_transform(df[['age', 'watch_time', 'days_since_login']])

df.to_csv("processed_customer_data.csv", index=False, encoding='utf-8-sig')  #csv파일로 저장
print("전처리 완료! 'processed_customer_data.csv' 파일로 저장됐습니다.")
print(df.head())
