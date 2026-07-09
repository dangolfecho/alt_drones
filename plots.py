import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

arr = np.genfromtxt('backup/results/sac/progress.csv', delimiter=',', names=True)

print(arr.dtype.names)
print(arr.shape)


#plt.plot(arr[:][11], arr[:][0])
