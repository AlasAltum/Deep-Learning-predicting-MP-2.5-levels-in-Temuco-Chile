"""Log Info. Stores info from a model to record it."""


class LogInfo():

    def __init__(self, model, logging_name):
        self.model = model
        self.training_info = []
        self.model_loss = {'train': 0, 'test': 0}
        self.training_set_size = 0
        self.training_time = 0.
        self.testing_set_size = 0
        self.commentary = ""
        self.logging_name = logging_name

    def set_comment(self, commentary):
        self.commentary = commentary

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

    def export_info(self):
        with open(f"./results/{self.logging_name}.log", 'w') as f:
            print(f'{str(self)}', file=f)

    def __str__(self):
        return f"""Model Name: {self.model.name}
        Training info: {self.training_info}
        Loss: {self.model_loss}
        Training Set Size: {self.training_set_size}
        Training time: {self.training_time}
        Testing Set Size: {self.testing_set_size}
        ------------------------------------------
        Additional Comments:
        {self.commentary}
        """

