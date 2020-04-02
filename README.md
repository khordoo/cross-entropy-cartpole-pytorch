# CartPole Cross-entropy Solver Agent

![Build Status](https://img.shields.io/static/v1?label=cartpole&message=cross-entropy&color=green)

This is a Pytorch agent that uses cross entropy RL method for solving the Cartpole environment. The idea is adopted from Maxim Lapan's book.
Changes:
- I refactored the code and introduced some new classes to make it easier to follow.
- Adjusted some hyperparameters for faster convergence.

# Deployment
- Install the dependencies and run the Jupyter notebook 

Note: To view the result of training steps in tensorboard perfrom the following steps:
- Install tensorboard and tensorboardX
- Run the code and wait until the end of the training 
- Execute the following code in the notebook
```sh
%load_ext tensorboard
%tensorboard --logdir runs
```

## Results:
The environment is relatively simple and converges very fast even on CPU. Here is the training results 

### Loss 
![image](https://user-images.githubusercontent.com/32692718/78293991-caaf2f00-74e6-11ea-84f7-ac619a0a96ae.png)

## Mean Reward
![image](https://user-images.githubusercontent.com/32692718/78294134-0cd87080-74e7-11ea-85ee-a452dbdc1ac1.png)

### Hyperparameters

The following hyperparameters needs are used and can be fine tuned:

| Parameter | Description |
| ------ | ------ |
| HIDDEN_SIZE | Number of hidden units in the linear layer |
| BATCh_SIZE | Batch size used during the training of NN |
| PERCENTILE | Percentile cut to select best performing episodes |
