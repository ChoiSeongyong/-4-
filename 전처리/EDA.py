# 탐색적 데이터 분석
import pandas as pd
import numpy as np
from datetime import datetime

# 1. 데이터 불러오기
df = pd.read_csv("dummy_customer_data.csv", encoding='utf-8-sig')

# 2. 결측치 확인
print("[1] 결측치 개수:")
missing = df.isnull().sum()
if missing.sum() == 0:
    print("→ 결측치 없음 ")
else:
    print(missing)

# 3. 수치 통계 요약
print("[2] 전체 평균 시청시간: {:.2f}분".format(df['watch_time'].mean()))
print("[3] 전체 평균 나이: {:.2f}세".format(df['age'].mean()))

# 4. 이탈 여부별 통계
df['churned'] = df['payment_status'].map({'paid': 0, 'unpaid': 1})
print("\n[4] 결제 상태별 고객 수:")
print(df['churned'].value_counts())

print("\n[5] 결제 상태별 평균 나이:")
print(df.groupby('churned')['age'].mean())

print("\n[6] 결제 상태별 평균 시청시간:")
print(df.groupby('churned')['watch_time'].mean())

# 5. 선호 장르 분포
print("\n[7] 선호 장르 분포:")
print(df['preferred_category'].value_counts())

# 6. 로그인 파생 변수 추가
df['last_login'] = pd.to_datetime(df['last_login'])
df['days_since_login'] = (datetime.today() - df['last_login']).dt.days
print("\n[8] 평균 로그인 경과 일 수: {:.2f}일".format(df['days_since_login'].mean()))
print("\n[9] 결제 상태별 평균 로그인 경과일:")
print(df.groupby('churned')['days_since_login'].mean())

# 7. 이상치 탐지
def detect_outliers_zscore(data, threshold=3):
    mean = np.mean(data)
    std = np.std(data)
    z_scores = [(x - mean) / std for x in data]
    return [i for i, z in enumerate(z_scores) if abs(z) > threshold]

age_outliers = detect_outliers_zscore(df['age'])
watch_outliers = detect_outliers_zscore(df['watch_time'])

print("\n[10] 이상치 탐지 결과 (Z-score ±3 기준):")
print("→ 나이 이상치 {}건: {}".format(len(age_outliers), age_outliers if age_outliers else "없음"))
print("→ 시청시간 이상치 {}건: {}".format(len(watch_outliers), watch_outliers if watch_outliers else "없음"))
