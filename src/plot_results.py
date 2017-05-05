import numpy as np
import matplotlib.pyplot as plt

with open('training_log.txt', 'r') as f:
    lines = f.readlines()
    lines = [l.strip('\r\n') for l in lines]
    lines = [l.split(',') for l in lines]
    lines = lines[1:]
    num_epochs = len(lines)
    train_err = 1 - np.array([l[1] for l in lines]).astype('float')
    val_err = 1 - np.array([l[3] for l in lines]).astype('float')

plt.figure()
plt.plot(np.arange(num_epochs), train_err)
plt.plot(np.arange(num_epochs), val_err, 'r--')
plt.legend(['train error', 'val error'])
plt.xlabel('epochs')
plt.ylabel('error')
plt.title('Inception_v3 Finetune Error')
plt.savefig('inception_food101.jpg')
