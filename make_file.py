import pandas as pd
from datetime import datetime

# 데이터 불러오기
df = pd.read_excel("dummy_customer_data.xlsx")

# last_login을 datetime으로 변환
df['last_login'] = pd.to_datetime(df['last_login'], errors='coerce')

# 기준 날짜 설정
today = datetime.today()

# 로그인 후 경과 일수 계산
df['days_since_login'] = (today - df['last_login']).dt.days

# 이탈자 정의: 30일 이상 로그인 안 한 경우
df['churned'] = df['days_since_login'].apply(lambda x: 1 if x >= 30 else 0)

# 필요시 저장
df.to_csv("labeled_customer_data.csv", index=False)
