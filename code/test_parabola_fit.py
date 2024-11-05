import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def parabola(x, a, b, c):
    return a * x**2 + b * x + c

#test data 
y_data = np.array([10, 8, 4, 5, 1, 3, 9, 12])
x_data = np.array([1, 2, 3, 4, 5, 6, 7, 8])

params, covariance = curve_fit(parabola, x_data, y_data)

a, b, c = params
print(f"Fitted parameters: a={a}, b={b}, c={c}")

plt.scatter(x_data, y_data, label='Data')
x_fit = np.linspace(min(x_data), max(x_data), 100)
y_fit = parabola(x_fit, a, b, c)
print("Min in y_fit: " , min (y_fit))
print("Index in y fit: ", np.argmin(y_fit) )
print("X value at that index: ", x_fit[np.argmin(y_fit)])
plt.plot(x_fit, y_fit, color='red', label='Fitted parabola')
plt.legend()
plt.show()