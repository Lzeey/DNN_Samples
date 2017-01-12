# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 21:41:27 2017

@author: Zeyi
"""

import numpy as np
import pandas as pd
import h2o
from h2o.estimators import H2ODeepLearningEstimator

if __name__ == "__main__":
    
    #Configurations: Size of array
    m = 2000
    n = 20
    
    #Generate a random array for autoencoder training
    rand_mat = np.random.randint(0, 100, m*n).reshape((m,n))
    
    #Connect to a h2o cluster
    h2o.init()
    h2o.removeAll() #Clean slate for cluster
    h2o_df = h2o.H2OFrame(python_obj=rand_mat)

    #Define our model
    model = H2ODeepLearningEstimator(activation='tanh',
                                    adaptive_rate=True,
                                    autoencoder=True,
                                    epochs=1000,
                                    hidden=[n/2, n/4, n/2],
                                    l2=0.05,
                                    loss='quadratic',
                                    mini_batch_size=100)
    
    #Train
    model.train(x=range(n), training_frame=h2o_df)
    
    #Retrieve the reconstruction error
    #error = model.r2().as_data_frame(use_pandas=True) #Bug in this now. Does not work
    #We perform the inference step to get our reconstruction error
    error = model.anomaly(test_data=h2o_df).as_data_frame(use_pandas=True)
