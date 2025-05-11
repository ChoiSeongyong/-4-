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

    for _ in range(n):
        name = fake.name()
        age = random.randint(18, 65)
        last_login = datetime.today() - timedelta(days=random.randint(0, 30))
        watch_time = random.randint(0, 300)
        category = random.choice(categories)
        email = fake.email()

        if watch_time < 60 and (datetime.today() - last_login).days > 10:
            payment = 'unpaid'
        else:
            payment = 'paid'

        data.append({
            'name': name,
            'age': age,
            'last_login': last_login.date().isoformat(),
            'watch_time': watch_time,
            'preferred_category': category,
            'payment_status': payment,
            'email': email
        })

    return pd.DataFrame(data)

# 실행
df = generate_dummy_data(500)
df.to_csv("dummy_customer_data.csv", index=False, encoding='utf-8-sig')  # CSV로 저장
print(df.head())  # 앞 5개만 확인/ print(df)하면 전체 결과 확인
