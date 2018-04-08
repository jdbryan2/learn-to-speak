
import os
import numpy as np
import pylab as plt

directory = "data/batch"

index_list = []  # using a list for simplicity
if os.path.exists(directory):
    for filename in os.listdir(directory):
        if filename.startswith('round') and filename.endswith(".npz"):
            index_list.append(int(filter(str.isdigit, filename)))


index_list = np.array(sorted(index_list))

error = np.array([])
for index in index_list:
    print index 
    fname = os.path.join(directory, 'round'+str(index)+'.npz')
    data = np.load(fname)
    if index == 1:
        old_F = np.zeros(data['F'].shape)

    error = np.append(error, np.sum(np.abs(old_F-data['F'])))
    old_F = np.copy(data['F'])
    

plt.plot(error)
plt.title("Change in Predictor Operator")
plt.ylabel("$ \sum | \Delta F |$")
plt.xlabel("Batch Count (60 sec each)")
plt.show()
