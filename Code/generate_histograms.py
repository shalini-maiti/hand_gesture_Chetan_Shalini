import numpy as np
import matplotlib.pyplot as plt

descriptors = np.load('descriptor_training_data.npy')
labels = np.load('descriptor_training_labels.npy')

print(descriptors.shape)

fig, ax = plt.subplots(4)
ax[0].hist(descriptors[0])
ax[1].hist(descriptors[200])
ax[2].hist(descriptors[400])
ax[3].hist(descriptors[700])
print(labels[0], labels[200], labels[400], labels[700])

'''
x = [value1, value2, value3,....]
plt.hist(x, bins = number of bins)
'''