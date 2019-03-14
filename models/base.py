from os import path


class BaseModel(object):

    def __init__(self, input_dim, n_classes, **kwargs):
        self.input_dim = input_dim
        self.n_classes = n_classes
        print("Input dim: {}".format(self.input_dim))

        # default arguments
        for name, value in self.defaults.items():
            setattr(self, name, kwargs.get(name, value))

        # check undefined arguments
        for name, value in self.defaults.items():
            if value is None:
                raise ValueError("Did not supply a value for {}.".format(name))

        # load weights if a valid checkpoint path is passed
        self.build_model()
        self.load(**kwargs)

    def save(self, checkpoint_path):
        # save function that saves the checkpoint in the path defined in the config file
        if self.model is None:
            raise Exception("You have to build the model first.")

        print("Saving model...")
        self.model.save_weights(checkpoint_path)
        print("Model saved")

    def load(self, **kwargs):
        # load latest checkpoint from the experiment path defined in the config file
        checkpoint_path = kwargs.get("load_checkpoint_file")
        if not checkpoint_path:
            return

        if self.model is None:
            raise Exception("You have to build the model first.")

        if not path.exists(checkpoint_path):
            raise IOError("Checkpoint file {} does not exist.".format(checkpoint_path))

        print("\nLoading model checkpoint {}".format(checkpoint_path))
        self.model.load_weights(checkpoint_path)

    def build_model(self):
        raise NotImplementedError
