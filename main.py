import joblib
import pandas as pd 
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
data=pd.read_csv("Iris.csv")
split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_idx,test_idx in split.split(data,data['Species']):
    strat_train_set=data.loc[train_idx]
    strat_test_set=data.loc[test_idx]


x_train=strat_train_set.drop(['Id','Species'],axis=1)
y_train=strat_train_set['Species']

x_test=strat_test_set.drop(['Id','Species'],axis=1)
y_test=strat_test_set['Species']

model=RandomForestClassifier(random_state=42)
model.fit(x_train,y_train)

y_pred=model.predict(x_test)


print("Accuracy:", accuracy_score(y_test, y_pred)*100)

results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

results.to_csv("iris_predictions.csv", index=False)
print("Predictions saved to iris_predictions.csv")

joblib.dump(model, 'iris_rf_model.pkl')
print("Model saved to iris_rf_model.pkl")



