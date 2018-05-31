import pickle
import matplotlib.pyplot as plt

path= "20_newsgroups/"
#filename = "results/comparison/CNN_optimization1/optimization_1_data/history_iter5.p"
filename = 'kmeans.p'
with open(path + "/" + filename, 'rb') as fp:
    history = pickle.load(fp)

plt.figure()
plt.plot(history)