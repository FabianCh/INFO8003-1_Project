import csv
import matplotlib.pyplot as plt

y_er_static, y_er_uniform, y_er_balanced = [], [], []
y_er_bias_static, y_er_bias_uniform, y_er_bias_balanced = [], [], []
y_h_static, y_h_uniform, y_h_balanced = [], [], []
y_h_bias_static, y_h_bias_uniform, y_h_bias_balanced = [], [], []
esup, eund = [], []

import os
print(os.getcwd())

with open("../output/evaluation2/Static150.csv",
          'r') as file:
    reader = csv.reader(file, delimiter=';')
    for row in reader:
        y_er_static.append(float(row[0]))
        y_er_bias_static.append(float(row[1])**(1/2))
        y_h_static.append(float(row[2]))
        y_h_bias_static.append(float(row[3])**(1/2))

with open("../output/evaluation2/U150.csv",
          'r') as file:
    reader = csv.reader(file, delimiter=';')
    for row in reader:
        y_er_uniform.append(float(row[0]))
        y_er_bias_uniform.append(float(row[1])**(1/2))
        y_h_uniform.append(float(row[2]))
        y_h_bias_uniform.append(float(row[3])**(1/2))
        esup.append(float(row[2])+(float(row[3])**(1/2)))
        eund.append(float(row[2]) - (float(row[3]) ** (1 / 2)))

with open("../output/evaluation2/BU150_2.csv",
          'r') as file:
    reader = csv.reader(file, delimiter=';')
    for row in reader:
        y_er_balanced.append(float(row[0]))
        y_er_bias_balanced.append(float(row[1])**(1/2))
        y_h_balanced.append(float(row[2]))
        y_h_bias_balanced.append(float(row[3])**(1/2))

plt.plot([i for i in range(len(y_er_static))], y_er_static, '#362893', label='Static Sampler')
plt.plot([i for i in range(len(y_er_uniform))], y_er_uniform, '#b32d30', label='Uniform Sampler')
plt.plot([i for i in range(len(y_er_balanced))], y_er_balanced, '#4d7e2b', label='Balanced Uniform Sampler')

plt.errorbar([i for i in range(len(y_er_static))], y_er_static, y_er_bias_static, fmt='none',  ecolor='#a0bbff', label='Static Sampler error' )
plt.errorbar([i for i in range(len(y_er_uniform))], y_er_uniform, y_er_bias_uniform, fmt='none', ecolor='#ffa8a8', label='Uniform Sampler error')
plt.errorbar([i for i in range(len(y_er_balanced))], y_er_balanced, y_er_bias_balanced, fmt='none', ecolor='#9cff58', label='Balanced Uniform Sampler error')

plt.legend(loc="upper left")
plt.xlabel("Episodes")
plt.ylabel("Expected Return")
plt.show()


plt.plot([i for i in range(len(y_h_static))], y_h_static, '#362893', label='Static Sampler')
plt.plot([i for i in range(len(y_h_uniform))], y_h_uniform, '#b32d30', label='Uniform Sampler')
plt.plot([i for i in range(len(y_h_balanced))], y_h_balanced, '#4d7e2b', label='Balanced Uniform Sampler')

plt.errorbar([i for i in range(len(y_h_static))], y_h_static, y_h_bias_static, fmt='none',  ecolor='#a0bbff', label='Static Sampler error' )
plt.errorbar([i for i in range(len(y_h_uniform))], y_h_uniform, y_h_bias_uniform, fmt='none', ecolor='#ffa8a8', label='Uniform Sampler error')
plt.errorbar([i for i in range(len(y_h_balanced))], y_h_balanced, y_h_bias_balanced, fmt='none', ecolor='#9cff58', label='Balanced Uniform Sampler error')

plt.legend(loc="upper left")
plt.xlabel("Episodes")
plt.ylabel("Number of hits")
plt.show()
