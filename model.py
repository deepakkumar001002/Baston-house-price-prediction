import pickle
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def train_and_save_model(filename='house_model.pkl'):
    data = load_boston()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print("MSE:", mean_squared_error(y_test, preds))
    print("R²:", r2_score(y_test, preds))
    with open(filename, 'wb') as f:
        pickle.dump((model, data.feature_names), f)

if __name__ == "__main__":
    train_and_save_model()
