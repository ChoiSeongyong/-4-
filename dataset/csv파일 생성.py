import pandas as pd
import random
from faker import Faker
from datetime import datetime, timedelta
import calendar

fake = Faker('ko_KR')
Faker.seed(42)

def generate_dummy_data(n=500):
    data = []

    categories = ['드라마', '예능', '다큐멘터리', '영화', '애니메이션']

    today = datetime.today()
    prev_month = today.month - 1 if today.month > 1 else 12
    prev_year = today.year if today.month > 1 else today.year - 1
    _, last_day = calendar.monthrange(prev_year, prev_month)
    criteria_date = datetime(prev_year, prev_month, last_day)
    
    for _ in range(n):
        name = fake.name()
        age = random.randint(18, 65)
        login_day = random.randint(1, last_day)
        last_login = datetime(prev_year, prev_month, login_day)
        watch_time = random.randint(0, 250)
        category = random.choice(categories)
        email = fake.email()

        if (criteria_date - last_login).days > 10 and watch_time < 35:
            payment_status = 'unpaid'
        else:
            payment_status = 'paid'

        data.append({
            'name': name,
            'age': age,
            'last_login': last_login.date().isoformat(),
            'watch_time': watch_time,
            'preferred_category': category,
            'payment_status': payment_status,
            'email': email
        })

    return pd.DataFrame(data)

# 실행
df = generate_dummy_data(500)
df.to_csv("dummy_customer_data.csv", index=False, encoding='utf-8-sig')  # CSV로 저장
print(df.head())  # 앞 5개만 확인/ print(df)하면 전체 결과 확인
