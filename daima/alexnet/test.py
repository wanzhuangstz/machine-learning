import numpy as np
import matplotlib.pyplot as plt
my_y_ticks = np.arange(0.5, 1, 0.05)
my_x_ticks = np.arange(0, 50, 2)

x=np.arange(0,20)
print(x)
y=0.05*x
plt.plot(x, y)
plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks)
plt.show()
