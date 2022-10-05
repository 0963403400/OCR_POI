from model_doi import Model

import tensorflow as tf
import numpy as np
import pdb
import os

def main():
    ### create model
    # tf.compat.v1.enable_eager_execution()
    # tf.compat.v1.enable_eager_execution(config=None, device_policy=None, execution_mode=None)
    model = Model()
    model.create_model_train()
    ### train new model
    model.train()

if __name__ == '__main__':
    main()
