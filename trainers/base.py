class BaseTrain(object):
    def __init__(self, description, model, **kwargs):

        # default arguments
        for name, value in self.defaults.items():
            setattr(self, name, kwargs.get(name, value))

        # check undefined arguments
        for name, value in self.defaults.items():
            if value is None:
                raise ValueError("Did not supply a value for {}.".format(name))

        self.description = description
        self.model = model

    def train(self):
        raise NotImplementedError
