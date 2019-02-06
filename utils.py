import os
import importlib


def create_dirs(dirs):
    """
    dirs - a list of directories to create if these directories are not found
    :param dirs:
    :return exit_code: 0:success -1:failed
    """
    for dir_ in dirs:
        if not os.path.exists(dir_):
            os.makedirs(dir_)
    return 0


def factory(cls):
    '''expects a string that can be imported as with a module.class name'''
    module_name, class_name = cls.rsplit(".", 1)

    somemodule = importlib.import_module(module_name)
    cls_instance = getattr(somemodule, class_name)

    return cls_instance
