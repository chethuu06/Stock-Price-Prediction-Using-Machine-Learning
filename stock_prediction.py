
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data = {
    'Day': [1,2,3,4,5,6,7],
    'Price': [100,102,101,105,107,110,112]
}

df = pd.DataFrame(data)
X = df[['Day']]
y = df['Price']

model = LinearRegression()
model.fit(X, y)

future_day = [[8]]
prediction = model.predict(future_day)
print("Predicted price for Day 8:", prediction[0])

plt.plot(df['Day'], df['Price'])
plt.scatter(8, prediction[0])
plt.show()
