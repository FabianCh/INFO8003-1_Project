import csv
import matplotlib.pyplot as plt
import numpy

y_er_static, y_er_uniform, y_er_balanced = [], [], []
y_h_static, y_h_uniform, y_h_balanced = [], [], []

with open("output/evaluation - Copie/Train--initialbuffersize_50000-episode_100-episode_size_1000-minibatchsize_30-decreaserate_0.00001-targetupdate_10000-Agent--agent_DDQNAgent-estimator_NeuralNetwork-buffer_Prioritized-maximizer_Static-Static--numberofsample_11-_log.csv",
          'r') as file:
    reader = csv.reader(file, delimiter=';')
    for row in reader:
        y_er_static.append(float(row[0]))
        y_h_static.append(float(row[1]))

with open("output/evaluation - Copie/Train--initialbuffersize_50000-episode_100-episode_size_1000-minibatchsize_30-decreaserate_0.00001-targetupdate_10000-Agent--agent_DDQNAgent-estimator_NeuralNetwork-buffer_Prioritized-maximizer_Uniform-Uniform--numberofsample_11-_log.csv",
          'r') as file:
    reader = csv.reader(file, delimiter=';')
    for row in reader:
        y_er_uniform.append(float(row[0]))
        y_h_uniform.append(float(row[1]))

with open("output/evaluation - Copie/Train--initialbuffersize_50000-episode_100-episode_size_1000-minibatchsize_30-decreaserate_0.00001-targetupdate_10000-Agent--agent_DDQNAgent-estimator_NeuralNetwork-buffer_Prioritized-maximizer_BalancedUniform-BalancedUniform--numberofsample_5-_log.csv",
          'r') as file:
    reader = csv.reader(file, delimiter=';')
    for row in reader:
        y_er_balanced.append(float(row[0]))
        y_h_balanced.append(float(row[1]))

plt.plot([i for i in range(len(y_er_static))], y_er_static, 'b', label='Static Sampler')
plt.plot([i for i in range(len(y_er_uniform))], y_er_uniform, 'r', label='Uniform Sampler')
plt.plot([i for i in range(len(y_er_balanced))], y_er_balanced, 'g', label='Balanced Uniform Sampler')
plt.legend(loc="upper left")
plt.xlabel("Episodes")
plt.ylabel("Expected Return")
plt.show()

plt.plot([i for i in range(len(y_h_static))], y_h_static, 'b', label='Static Sampler')
plt.plot([i for i in range(len(y_h_uniform))], y_h_uniform, 'r', label='Uniform Sampler')
plt.plot([i for i in range(len(y_h_balanced))], y_h_balanced, 'g', label='Balanced Uniform Sampler')
plt.legend(loc="upper left")
plt.xlabel("Episodes")
plt.ylabel("Number of hits")
plt.show()