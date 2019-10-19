import time

import numpy as np
import tensorflow as tf

import util.dataloader as dataloader
from util.config import Config

"""Benchmark for building a RNN model.
The hyperparameters used in the model:
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of RNN layers
- num_steps - the number of unrolled steps of RNN
- hidden_size - the number of RNN units
- epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate
- batch_size - the batch size
"""

RNN = 'rnn'
LSTM = 'lstm'
GRU = 'gru'

flags = tf.flags
flags.DEFINE_string("config_file", None, "Model config file")
FLAGS = flags.FLAGS

class ModelInput():

    def __init__(self,  raw_data, batch_size):
        input_data, target, data_iterator, iteration = self._producer(raw_data,
                                                                     batch_size)
        self.input_data = input_data
        self.target = target
        self.data_iterator = data_iterator
        self.iteration = iteration

    def _producer(self, raw_data, batch_size):
        iteration = int(len(raw_data) / batch_size) + 1
        data = tf.convert_to_tensor(raw_data, dtype=tf.int64)

        dataset = tf.data.Dataset.from_tensor_slices((data[:,0,:], data[:,1,:]))
        dataset = dataset.shuffle(data.shape[0]).batch(batch_size)
        iterator = dataset.make_initializable_iterator()
        input_, output_ = iterator.get_next()
        return input_, output_, iterator, iteration


class Model():

    def __init__(self, is_training, config, input_):
        self._batch_size = config.batch_size
        self._num_steps = config.num_steps
        self._hidden_size = config.hidden_size
        self._vocab_size = config.vocab_size
        self._label_size = config.label_size
        self._num_layers = config.num_layers
        self._input = input_
        self._model_type = config.model

        self._keep_prob = tf.placeholder_with_default(1.0, shape=[], name="keep_prob")

        embedding = tf.get_variable("embedding",
                                    [self._vocab_size, self._hidden_size],
                                    dtype=tf.float32)
        inputs = tf.nn.embedding_lookup(embedding, self._input.input_data)

        inputs = tf.nn.dropout(inputs, self.keep_prob)

        output = self._build_rnn_graph(inputs, self._num_layers,
                                       self._hidden_size,
                                       self._num_steps,
                                       self._model_type)
        softmax_w = tf.get_variable("softmax_w",
                                    [self._hidden_size, self._label_size],
                                    dtype=tf.float32)

        softmax_b = tf.get_variable("softmax_b", [self._label_size],
                                    dtype=tf.float32)
        logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)

        # Reshape logits to be a 3-D tensor for sequence loss
        logits = tf.reshape(logits, [-1, self._num_steps,
                                     self._label_size])

        predict = tf.nn.softmax(logits)
        predict = tf.reshape(predict, [-1, self._num_steps,
                                       self._label_size])
        self._predict = tf.argmax(predict, 2, name="predict")

        bool_label_compare = tf.equal(self._predict, self._input.target)
        fl_label_compare = tf.cast(bool_label_compare, tf.float32)
        fl_senten_compare = tf.reduce_min(fl_label_compare, axis=1)
        self._label_accuracy = tf.reduce_mean(fl_label_compare)
        self._sentence_accuracy = tf.reduce_mean(fl_senten_compare)

        # compute loss
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self._input.target
                                                              , logits=logits)
        batch_avg_loss = tf.reduce_mean(loss, 0)

        # Update the cost
        self._cost = tf.reduce_sum(batch_avg_loss, name="loss")

        if not is_training:
            return

        self._lr = tf.Variable(0.0, trainable=False)

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, tvars),
                                          config.max_grad_norm)

#         optimizer = tf.train.GradientDescentOptimizer(self._lr)
        optimizer = tf.train.AdamOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.train.get_or_create_global_step())

        self._new_lr = tf.placeholder(tf.float32, shape=[],
                                      name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

        tf.summary.scalar("Training Loss", self._cost)
        tf.summary.scalar("Learning Rate", self._lr)
        tf.summary.scalar("Label accuracy", self._label_accuracy)
        tf.summary.scalar("Sentence accuracy", self._sentence_accuracy)

        summary_op = tf.summary.merge_all()
        self._summary_op = summary_op

    def _get_lstm_cell(self, hidden_size, model_type):
        if model_type == GRU:
            cell = tf.contrib.rnn.GRUCell(hidden_size)
        elif model_type == LSTM:
            cell = tf.contrib.rnn.BasicLSTMCell(hidden_size,
                                                    forget_bias=1.0,
                                                    state_is_tuple=True)
        else:
            cell = tf.contrib.rnn.BasicRNNCell(hidden_size)
        cell = tf.contrib.rnn.DropoutWrapper(cell,
                                                  output_keep_prob=self.keep_prob)
        return cell

    def _build_rnn_graph(self, inputs, num_layers, hidden_size, num_steps, model_type):
        cell = tf.contrib.rnn.MultiRNNCell(
            [self._get_lstm_cell(hidden_size, model_type) for _ in range(num_layers)], state_is_tuple=True)
        inputs = tf.unstack(inputs, num=num_steps, axis=1)
        outputs, state = tf.nn.static_rnn(cell, inputs, dtype=tf.float32)
        output = tf.reshape(tf.concat(outputs, 1), [-1, hidden_size])
        return output

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    def export_ops(self):
        ops = {"predict": self.predict, "cost": self.cost,
               "keep_prob": self.keep_prob}
        for name, op in ops.items():
            tf.add_to_collection(name, op)

    @property
    def input(self):
        return self._input

    @property
    def keep_prob(self):
        return self._keep_prob

    @property
    def predict(self):
        return self._predict

    @property
    def label_accuracy(self):
        return self._label_accuracy

    @property
    def sentence_accuracy(self):
        return self._sentence_accuracy

    @property
    def summary_op(self):
        return self._summary_op

    @property
    def cost(self):
        return self._cost

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op


def run_epoch(session, model, config, is_training, writer=None, epoch=0):
    costs = 0.0
    label_accuracy = 0.0
    senten_accuracy = 0.0
    step = 0
    iteration = model.input.iteration
    predictions = []
    targets = []

    try:
        while True:
            feed_dict = dict()
            feed_dict[model.keep_prob] = config.keep_prob

            if is_training:
                cost, label_acc, sen_acc, summary, _ = session.run(
                    [model.cost, model.label_accuracy, model.sentence_accuracy,
                     model.summary_op, model.train_op],
                    feed_dict)
                writer.add_summary(summary,
                                   global_step=epoch * iteration + step)
            else:
                cost, label_acc, sen_acc, pred, target = session.run(
                    [model.cost, model.label_accuracy, model.sentence_accuracy,
                     model.predict, model.input.target])
                predictions.append(pred)
                targets.append(target)

            costs += cost
            # accuracy da chia cho num_steps va batch_size
            label_accuracy += label_acc
            senten_accuracy += sen_acc
            # loss da chia cho batch_size va lay tong cac num_steps
            elements = config.num_steps * (step + 1)

            step += 1

    except tf.errors.OutOfRangeError:
        pass

    cost_avg = costs / (config.num_steps * iteration)
    label_accuracy = label_accuracy / iteration
    senten_accuracy = senten_accuracy / iteration

    if not is_training:
        predict_np = np.asarray(predictions)
        predict_np = np.concatenate(predict_np, axis=0)
        target_np = np.asarray(targets)
        target_np = np.concatenate(target_np, axis=0)
        macro_avg_score = f1(config, predict_np, target_np)

        return cost_avg, label_accuracy, senten_accuracy, macro_avg_score
    else:
        return cost_avg, label_accuracy, senten_accuracy


def f1(config, prediction, target):
    max_length = config.num_steps

    tp = np.zeros(config.label_size)
    fp = np.zeros(config.label_size)
    fn = np.zeros(config.label_size)

    for i in range(len(target)):
        for j in range(max_length):
            if target[i, j] == prediction[i, j]:
                tp[target[i, j]] += 1.0
            else:
                fp[target[i, j]] += 1.0
                fn[prediction[i, j]] += 1.0

    precision = []
    recall = []
    fscore = []
    for i in range(config.label_size):
        if fp[i] == 0.0:
            precision.append(1.0)
        else:
            precision.append(tp[i] / (tp[i] + fp[i]))

        if fn[i] == 0.0:
            recall.append(1.0)
        else:
            recall.append(tp[i] / (tp[i] + fn[i]))

        fscore.append(
            2.0 * precision[i] * recall[i] / (precision[i] + recall[i]))

    precision_avg = np.mean(np.asarray(precision[1:]))
    recall_avg = np.mean(np.asarray(recall[1:]))
    macro_avg_score = 2.0 * precision_avg * recall_avg / (
                precision_avg + recall_avg)
    return macro_avg_score


def train(sess, model, test_model, config):
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(config.save_path + "/summary", sess.graph)

    start = time.time()
    print(r"""{"result":[[""", end="")
    for epoch in range(config.epoch):
        sess.run(model.input.data_iterator.initializer)
        lr_decay = 1 / (1 + config.lr_decay * epoch)
        model.assign_lr(sess, config.learning_rate * lr_decay)
        lr = sess.run(model.lr)

        cost, label_accu, senten_accu = run_epoch(sess, model, config,
                                                  is_training=True,
                                                  writer=writer, epoch=epoch)
        sess.run(test_model.input.data_iterator.initializer)
        _, _, _, macro_score = run_epoch(sess, test_model, config,
                                         is_training=False)

        now = time.time()
        if epoch == config.epoch - 1:
            print("[{},{:0.3},{}]".format(epoch+1, macro_score, now-start), end="")
        else:
            print("[{},{:0.3},{}],".format(epoch+1, macro_score, now-start), end="")
        if config.save_path and (epoch + 1) % 10 == 0:
            saver.save(sess, config.save_path + "/saver", global_step=epoch + 1)

    print("]", end="")


def test(sess, model, config):
    sess.run(model.input.data_iterator.initializer)
    cost, label_accu, senten_accu, macro_score = run_epoch(sess, model, config,
                                                           is_training=False)
    print(",{:0.3}".format(macro_score), end="")

def main(_):

    if not FLAGS.config_file:
        raise ValueError("Must set --config_file to set model's hyperparams")

    config = Config(FLAGS.config_file)

    data = dataloader.Dataloader(config.data_file)
    dataset = data.load_data()

    config.num_steps = data.sentence_max_len
    config.vocab_size = data.vocab_size
    config.label_size = data.label_size
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)

    kfold = config.kfold
    total_data = len(dataset)
    for i in range(kfold):
        test_data = dataset[int(total_data / kfold * i):int(
            total_data / kfold * (i + 1))]
        train_data = dataset[0:int(total_data / kfold * i)]
        train_data.extend(dataset[int(total_data / kfold * (i + 1)):total_data])

        tf.reset_default_graph()
        with tf.name_scope("Train"):
            train_input = ModelInput(raw_data=train_data,
                                     batch_size=config.batch_size)
            with tf.variable_scope("Model", reuse=None,
                                   initializer=initializer):
                train_model = Model(is_training=True, config=config,
                                    input_=train_input)

        with tf.name_scope("Test"):
            test_input = ModelInput(raw_data=test_data,
                                    batch_size=config.batch_size)
            with tf.variable_scope("Model", reuse=True,
                                   initializer=initializer):
                test_model = Model(is_training=False, config=config,
                                   input_=test_input)

        with tf.name_scope("Test_train"):
            test_train_input = ModelInput(raw_data=train_data,
                                          batch_size=config.batch_size)
            with tf.variable_scope("Model", reuse=True,
                                   initializer=initializer):
                test_train_model = Model(is_training=False, config=config,
                                         input_=test_train_input)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            train(sess, train_model, test_model, config)
            test(sess, test_model, config)
            test(sess, test_train_model, config)
            print("""]}""")


if __name__ == "__main__":
    tf.app.run()
