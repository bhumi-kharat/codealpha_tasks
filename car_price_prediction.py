import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Sample dataset
data = {
    "Year": [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2014, 2013, 2012],
    "Present_Price": [5.59, 6.0, 7.5, 8.0, 9.5, 10.0, 11.0, 4.5, 3.5, 3.0],
    "Kms_Driven": [27000, 30000, 20000, 15000, 10000, 5000, 4000, 35000, 40000, 45000],
    "Selling_Price": [3.35, 4.0, 5.0, 6.0, 7.5, 8.0, 9.0, 2.5, 2.0, 1.8]
}

df = pd.DataFrame(data)

X = df[["Year", "Present_Price", "Kms_Driven"]]
y = df["Selling_Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("R2 Score:", r2_score(y_test, y_pred))
input("Press Enter to exit...")