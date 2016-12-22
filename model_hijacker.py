from __future__ import print_function

def hijack_model(model):

    def fit(*args, **kwargs):
        return

    model.__fit = model.fit
