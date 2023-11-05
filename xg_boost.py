import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

data = pd.read_csv('wfp_food_prices_kaz.csv')

data['year'] = pd.to_datetime(data['date']).dt.year
data['month'] = pd.to_datetime(data['date']).dt.month
data['day'] = pd.to_datetime(data['date']).dt.day

data = pd.get_dummies(data, columns=['market', 'commodity', 'date'])
X = data.drop('price', axis=1)
y = data['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=68)

model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=5,
                         learning_rate=0.2, random_state=84)
model.fit(X_train._get_numeric_data(), y_train)

y_pred = model.predict(X_test._get_numeric_data())

mse = mean_squared_error(y_test, y_pred)
print(f"Среднеквадратичная ошибка: {round(mse, 2)}")

new_data = pd.DataFrame({'market': ['Nur-Sultan', 'Kostanay'], 'commodity': ['Potatoes', 'Milk'], 'year': [2020, 2020], 'month': [12, 12], 'day': [15, 15]})  # Замените на свои данные
new_data = pd.get_dummies(new_data)

cols_when_model_builds = model.get_booster().feature_names
new_data = new_data.reindex(columns=cols_when_model_builds, fill_value=0)

predicted_price = model.predict(new_data)
print(f"Предсказанная цена Картошки в Астана на 2020-12-15, месяц спустя(последняя цена в датасете 2020-11-15: 124тг): {round(float(predicted_price[0]), 2)}")
print(f"Предсказанная цена Молока в Костанае на 2020-12-15, месяц спустя(последняя цена в датасете 2020-11-15: 166тг): {round(float(predicted_price[1]), 2)}")


