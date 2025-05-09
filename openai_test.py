import pandas as pd
import openai
import os
from datetime import datetime

# 환경 변수에서 OPENAI API 키를 가져옴
openai.api_key = os.environ.get("OPENAI_API_KEY")

# 고위험 고객의 데이터 가져옴
df = pd.read_csv('highrisk.csv')

current_date = datetime.now()

# csv파일을 행별로 가져와 고객의 특성 추출
for index, customer in df.iterrows():
    age = customer['age']
    preferred_category = customer['preferred_category']
    last_login = pd.to_datetime(customer['last_login'])
    email = customer['email']

    days_since_last_login = (current_date - last_login).days

    # GPT-3 모델을 이용해 맞춤형 메시지 생성
    prompt = f"""
    
    고객 데이터:
    - 고객님 나이: {age}세 (나이는 메시지에서 언급하면 안됨)
    - 고객님 선호 카테고리: {preferred_category}
    - 고객님의 마지막 로그인 후 {days_since_last_login}일이 지남.

    고객 데이터 기반 고객님에게 다음과 같은 내용을 포함하는 메시지 생성:
    1. 고객의 이름, 마지막 로그인 후 지난일 수를 언급하면서 걱정한 듯한 메시지
    2. 고객의 나이와 선호 카테고리 기반 고객 맞춤형 추천 리스트 3가지 추천 메시지
    3. 고객으로부터 서비스 이용을 유도할 수 있는 메시지와 지금 시정하러가기(넷플릭스 홈피로 이동) 클릭할 수 있는 메세지
    
    """

    # GPT-3 모델로 메시지 생성
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # 사용할 모델
        messages=[
            {"role": "system", "content": "고객 이탈을 막기 위한 맞춤형 메시지 생성."},
            {"role": "user", "content": prompt}
        ]
    )

    churn_message = response['choices'][0]['message']['content'].strip()
    print(f"이메일: {email}\n생성된 메시지:\n{churn_message}\n")

#추후 이메일 연동해서 아래에 코드 추가
