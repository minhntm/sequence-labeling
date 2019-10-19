import configparser

class Config():
    def __init__(self, config_file=None):
        if config_file is None:
            self.init_scale = 0.05
            self.learning_rate = 1.0
            self.max_grad_norm = 5
            self.num_layers = 1
            self.hidden_size = 100
            self.epoch = 2
            self.keep_prob = 0.8
            self.lr_decay = 0.5
            self.batch_size = 64
            self.save_path = 'checkpoint/saver'
            self.data_file = "../data/data.txt"
            self.model = "rnn"
            self.kfold = 10
        else:
            self._parse_config(config_file)

    def _parse_config(self, config_file):
        config = configparser.ConfigParser()
        section = "config"

        config.read(config_file)
        self.init_scale = float(config.get(section, "init_scale"))
        self.learning_rate = float(config.get(section, "learning_rate"))
        self.max_grad_norm = int(config.get(section, "max_grad_norm"))
        self.num_layers = int(config.get(section, "num_layers"))
        self.hidden_size = int(config.get(section, "hidden_size"))
        self.epoch = int(config.get(section, "epoch"))
        self.keep_prob = float(config.get(section, "keep_prob"))
        self.lr_decay = float(config.get(section, "lr_decay"))
        self.batch_size = int(config.get(section, "batch_size"))
        self.save_path = config.get(section, "save_path")
        self.data_file = config.get(section, "data_file")
        self.model = config.get(section, "model")
        self.kfold = int(config.get(section, "kfold"))
