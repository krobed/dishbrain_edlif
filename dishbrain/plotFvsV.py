import matplotlib.pyplot as plt
import numpy as np


x = np.linspace(0,5,1000)
plt.plot(x, np.exp(-x)*3 +2)
plt.title('Firing rate vs Velocity')
plt.ylabel('Velocity')
plt.xlabel('Firing rate')
plt.legend()
plt.grid()
plt.show()