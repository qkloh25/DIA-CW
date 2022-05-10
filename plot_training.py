import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd
with open('DQN_centralised_train_score', 'r') as f:
    data_head = json.load(f)

with open('DQN_distributed_train_score', 'r') as f:
    data2_head = json.load(f)


with open('DQN_centralised_train_score1', 'r') as f:
    data = json.load(f)

with open('DQN_distributed_train_score1', 'r') as f:
    data2 = json.load(f)
data = data_head + data 
data2 = data2_head + data2

data = pd.DataFrame(data)
data2 = pd.DataFrame(data2)
data2 = data2.head(5000000)

test = data[0].rolling(200).mean()
test2 = data2[0].rolling(200).mean()
plt.plot(test2)
plt.plot(test)
plt.legend(['Independent 2-DQN','One DQN' ])
plt.show()