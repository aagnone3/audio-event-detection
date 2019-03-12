from keras.utils import Sequence


class BaseDataLoader(Sequence):

    def __init__(self, **kwargs):
        # default arguments
        for name, value in self.defaults.items():
            setattr(self, name, kwargs.get(name, value))

        # check undefined arguments
        for name, value in self.defaults.items():
            if getattr(self, name) is None:
                raise ValueError("Did not supply a value for {}.".format(name))

    def get_train_data(self):
        raise NotImplementedError

    def get_test_data(self):
        raise NotImplementedError
