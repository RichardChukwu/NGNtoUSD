from sklearn.metrics import mean_absolute_error

with open("./artifacts/support_model.pkl", "rb") as model_file:
    y_preds = support_model.predict(X_test)
mae = mean_absolute_error(y_test, y_preds)
mae
