import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd
with open('DQN_centralised_evaluate', 'r') as f:
    data = json.load(f)

with open('DQN_distributed_evaluate', 'r') as f:
    data2= json.load(f)

data = pd.DataFrame(data)
data2 = pd.DataFrame(data2)

test = data[0].rolling(200).mean()
test2 = data2[0].rolling(200).mean()
plt.plot(test)
plt.plot(test2)
plt.legend(['One DQN','Independent 2-DQN'])
plt.show()