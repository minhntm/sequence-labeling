# Sequence-labeling
Sequence Labeling implemented by RNN/GRU/LSTM using Tensorflow.

# Install requirements
- tensorflow 1.6
- numpy
- python 3.6

# Model hyperparameters
- init_scale: the initial value of parameters
- learning_rate: the initial value of learning rate
- max_grad_norm: the maximum norm value of gradient
- num_layers: the layer number of model
- hidden_size: the node number of hidden layer
- epoch: the the total number of epochs for training
- keep_prob: the dropout probability
- lr_decay: the decay of the learning rate
- batch_size: the number of inputs
- model: model type (rnn/gru/lstm)
- save_path: the folder to save parameter after training
- data_file: data of model
- kfold: the k-fold value of cross-validation test method

# Evaluation
- kfold cross-validation method (*warning: because of extremely long running time, in deep learning area, we don't use kfold cross-validation to evaluate model. This is just a funny experiment.*)
- f1 score metric

# Training and evaluate model
```
python rnn_model.py --config_file=config.cfg
```