"""Log Info. Stores info from a model to record it."""

class LogInfo():

    def __init__(self, model, logging_name):
        self.model = model
        self.training_info = []
        self.model_loss = {'train': 0, 'test': 0}
        self.avg_loss = {'train': 0, 'test': 0}
        self.training_set_size = 0
        self.training_time = 0.
        self.testing_set_size = 0
        self.commentary = ""
        self.epochs = 0
        self.optimizer = "Adam"
        self.loss_function = "MSELoss"
        self.logging_name = logging_name

    def set_comment(self, commentary):
        self.commentary = commentary

    def set_epochs(self, epochs):
        self.epochs = epochs

    def add_info(self, info):
        self.commentary += info + '\n'

    def add_model_loss(self, t_set, loss):
        self.model_loss[t_set] = loss

    def set_train_test_size(self, x_train, x_test):
        try:
            self.training_set_size = x_train.shape
            self.testing_set_size = x_test.shape

        except:
            self.training_set_size = len(x_train)
            self.testing_set_size = len(x_test)

    def set_training_time(self, time):
        self.training_time = time

    def set_optimizer(self, optim):
        self.optimizer = type(optim).__name__

    def set_loss_function(self, loss_fn):
        self.loss_function = type(loss_fn).__name__

    def export_info(self):
        with open(f"./results/{self.logging_name}.log", 'w') as f:
            print(f'{str(self)}', file=f)

    def __str__(self):
        try:
            self.avg_loss['train'] = \
                self.model_loss['train'] / (self.training_set_size[0])
            self.avg_loss['test'] = \
                self.model_loss['test'] / (self.testing_set_size[0])

        except:
            self.avg_loss['train'] = \
                self.model_loss['train'] / (self.training_set_size)
            self.avg_loss['test'] = \
                self.model_loss['test'] / (self.testing_set_size)

        return (f'Model Name: {self.model.name} \n'
                f'Loss: {self.model_loss}\n'
                f'Average Loss: {self.avg_loss}\n'
                f'Optimizer: {self.optimizer}\n'
                f'Loss function: {self.loss_function}\n'
                f'# Epochs: {self.epochs}\n'
                f'Training time: {self.training_time:.1f}\n'
                f'Training Set Size: {self.training_set_size}\n'
                f'Testing Set Size: {self.testing_set_size}\n'
                f'------------------------------------------\n'
                f'Additional Comments:\n'
                f'{self.commentary}\n'
                )
