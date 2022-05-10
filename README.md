
# Demo Video
https://youtu.be/53Z-DgA-s6s

# Folder structure
 ```
📦DIA CW
 ┣ 📂tmp             # To save the weight of the neural network
 ┃ ┣ 📂DQN_agent_centralised
 ┃ ┃ ┣ 📜policy
 ┃ ┃ ┗ 📜target
 ┃ ┣ 📂DQN_agent_distributed_1
 ┃ ┃ ┣ 📜policy
 ┃ ┃ ┗ 📜target
 ┃ ┣ 📂DQN_agent_distributed_2
 ┃ ┃ ┣ 📜policy
 ┃ ┃ ┗ 📜target
 ┃ ┣ 📂MAA2C_agent_1
 ┃ ┗ 📂MAA2C_agent_2
 ┣ 📜agents.py
 ┣ 📜buffers.py
 ┣ 📜csettings.py
 ┣ 📜DQN_centralised_evaluate      #Evaluation score for 1-DQN
 ┣ 📜DQN_centralised_train_score   #Train score for 1-DQN
 ┣ 📜DQN_centralised_train_score1  #Train score for 1-DQN(more)
 ┣ 📜DQN_distributed_evaluate      #Evaluation score for 2-DQN
 ┣ 📜DQN_distributed_train_score   #Train score for 2-DQN
 ┣ 📜DQN_distributed_train_score1  #Train score for 1-DQN(more)
 ┣ 📜enviroments.py
 ┣ 📜evaluate_centralised.py    # Evaluate the 1DQN 
 ┣ 📜evaluate_distributed.py    # Evaluate the 2DQN
 ┣ 📜networks.py
 ┣ 📜plot_evaluate.py       # Plot the evaluation score
 ┣ 📜plot_training.py       # Plot the training score
 ┣ 📜README.md
 ┣ 📜train_DQN_centralised.py   # Train the 1DQN
 ┣ 📜train_DQN_distributed.py   # Train the 2DQN
 ┗ 📜train_MAA2C.py
 ```

 The files to visualise the results would be evaluate_centralised.py and evaluate_distributed.py. If you dont want to see the game running, set render = False.