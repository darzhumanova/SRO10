from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Генерируем случайные данные для примера
np.random.seed(42)
X = np.random.rand(100, 1)
y = 2 * X.squeeze() + 1 + 0.1 * np.random.randn(100)

# Разбиваем данные на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Определение базовых моделей
base_models = [
    ('linear_reg', LinearRegression()),
    ('decision_tree', DecisionTreeRegressor(random_state=42))
]

# Определение мета-модели (в данном случае, линейная регрессия)
meta_model = LinearRegression()

# Создание стекинга
stacking_regressor = StackingRegressor(estimators=base_models, final_estimator=meta_model)

# Обучение моделей
for model_name, model in base_models:
    model.fit(X_train, y_train)

stacking_regressor.fit(X_train, y_train)

# Предсказание на тестовых данных
base_models_predictions = np.column_stack([model.predict(X_test) for _, model in base_models])
stacking_predictions = stacking_regressor.predict(X_test)

# Оценка качества предсказания
for i, (model_name, _) in enumerate(base_models):
    mse = mean_squared_error(y_test, base_models_predictions[:, i])
    print(f'{model_name} MSE: {mse}')

stacking_mse = mean_squared_error(y_test, stacking_predictions)
print(f'Stacking MSE: {stacking_mse}')
