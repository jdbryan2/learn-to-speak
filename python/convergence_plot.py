
import os
import numpy as np
import pylab as plt

from plot_functions import *

tikz_dir = '/home/jacob/Projects/Dissertation/Doc/tikz/'

#directory = "data/batch_zeros_100_10"
directory = "data/batch_random_12_12"
#directory = "data/batch_random_20_10"

index_list = []  # using a list for simplicity
if os.path.exists(directory):
    for filename in os.listdir(directory):
        if filename.startswith('round') and filename.endswith(".npz"):
            index_list.append(int(filter(str.isdigit, filename)))


index_list = np.array(sorted(index_list))

error = np.array([])
mean_F = np.array([])
var_F = np.array([])

for index in index_list:
    print index 
    fname = os.path.join(directory, 'round'+str(index)+'.npz')
    data = np.load(fname)
    if index == 1:
        old_F = np.zeros(data['F'].shape)

    
    error = np.append(error, np.sum(np.abs(old_F-data['F'])/np.sum(np.abs(data['F']))))#/data['F'].size)
    #error = np.append(error, np.sum(np.abs(old_F-data['F']))/data['F'].size)
    old_F = np.copy(data['F'])
    mean_F = np.append(mean_F, np.mean(data['F']))
    var_F = np.append(var_F, np.var(data['F']))
    #plt.figure()
    #plt.imshow(np.abs(data['F']))
    #if index % 9 == 0:
        #plt.show()
    

plt.plot(np.log(error))
plt.title("Log Average Change in Prediction Operator")
plt.ylabel("$\Delta F$") #"$ \sum | \Delta F |$")
plt.xlabel("Batch Count (60 sec each)")
tikz_save(tikz_dir+'log_change_operator.tikz')

plt.figure()
plt.plot(100.*error)
plt.title("Percent Change in Prediction Operator")
plt.ylabel("100% * $\\frac{\Vert\Delta F\Vert}{\Vert F\Vert}$") #"$ \sum | \Delta F |$")
plt.xlabel("Batch Count (60 sec each)")
tikz_save(tikz_dir+'percent_change_operator.tikz')

plt.figure()
plt.plot(mean_F)
plt.title("Average value of element in F")
plt.ylabel("$F$") #"$ \sum | \Delta F |$")
plt.xlabel("Batch Count (60 sec each)")
tikz_save(tikz_dir+'average_operator_value.tikz')

plt.figure()
plt.plot(var_F)
plt.title("Variance of element in F")
plt.ylabel("$F$") #"$ \sum | \Delta F |$")
plt.xlabel("Batch Count (60 sec each)")
tikz_save(tikz_dir+'operator_variance.tikz')

plt.show()
