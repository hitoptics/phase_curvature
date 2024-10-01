##  CNN using Functional API & CoordEmb
##       Shn-ya Hasegawa
##
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'True'
import gc
import time
import pandas as pd
import numpy as np
import random
from numpy.random import uniform
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import statistics
import matplotlib.pyplot as plt

### tensorflow ###
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, BatchNormalization, Activation
from keras.layers.pooling import MaxPooling1D
from keras.layers.convolutional import UpSampling1D
from tensorflow import keras
from tensorflow.keras import Input, layers
from tensorflow.keras.layers import Add
from keras import regularizers
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger

gc.collect()

ran=1

tf.random.set_seed(ran)
for ks in range(1):
    q=ks

    tf.keras.utils.set_random_seed(146)   
    tf.config.experimental.enable_op_determinism()

    for xi in range(ran):
 
                    xi=xi+1 

                
                    def create_data():
                        
                        k_data = 'random_data_5%noise_csaps_400mic.csv'
                        L_data = 'random_label_5%noise_csaps_400mic.csv'                       
                        X = pd.read_csv(k_data) 
                        
                        Yr = pd.read_csv(L_data, usecols=[0,0])
                        Yn = pd.read_csv(L_data, usecols=[1,1])
                        Yz = pd.read_csv(L_data, usecols=[2,2])
                        Ydz = pd.read_csv(L_data, usecols=[3,3])

                        X = np.array(X)
                        Yr = np.array(Yr)
                        Yn = np.array(Yn)
                        Yz = np.array(Yz)
                        Ydz = np.array(Ydz)
                        
                        return X, Yr, Yn, Yz, Ydz
                    
                    if ks==0 and xi==1:
                        wran=random.sample(range(200), k=ran+1)
                                              
                    if xi==1: 
                        X, Yr, Yn, Yz, Ydz = create_data()
                        #X, Yr = create_data()
                    
                        Xmax =np.max(X)
                        Xmin = np.min(X)
                        Yrmax = np.max(Yr)
                        Yrmin = np.min(Yr)
                        Ynmax = np.max(Yn)
                        Ynmin = np.min(Yn)
                        Yzmax = np.max(Yz)
                        Yzmin = np.min(Yz)
                        Ydzmax = np.max(Ydz)
                        Ydzmin = np.min(Ydz)

                        X_train, X_test, Yr_train, Yr_test, Yn_train, Yn_test, Yz_train, Yz_test, Ydz_train, Ydz_test = train_test_split(X, Yr, Yn, Yz, Ydz, test_size=0.3, random_state=wran[xi])
                        
                        testdata=X_test
                        testr=Yr_test
                        testn=Yn_test 
                        testz=Yz_test
                        testdz=Ydz_test

                        print("X_test:", testdata.shape, "tesr:", testr.shape, "tesn:", testn.shape, "tesz:", testz.shape, "tesdz:", testdz.shape)
                        
                        X_test= np.array(testdata)
                        Yr_test=np.array(testr)                       
                        Yn_test=np.array(testn)
                        Yz_test=np.array(testz) 
                        Ydz_test=np.array(testdz)                        
                        
                        X_train = (X_train-Xmin)/(Xmax-Xmin)
                        X_test = (X_test-Xmin)/(Xmax-Xmin)
                        
                        h0=X_train.shape[0]
                        w0=X_train.shape[1]
                        print(h0)
                        coor1=np.zeros((h0,w0))
                        for j in range(w0):
                              coor1[:,j]=j/(w0-1)
                        X_train=(X_train-2*coor1)/3
                                                            
                        h1=X_test.shape[0]
                        w1=X_test.shape[1]
                        coor2=np.zeros((h1,w1))
                        for j in range(w1):
                              coor2[:,j]=j/(w1-1)
                        X_test=(X_test-2*coor2)/3
                       
                        Yr_train = (Yr_train-Yrmin)/(Yrmax-Yrmin)
                        Yr_test = (Yr_test-Yrmin)/(Yrmax-Yrmin)
                        Yn_train = (Yn_train-Ynmin)/(Ynmax-Ynmin)                    
                        Yn_test = (Yn_test-Ynmin)/(Ynmax-Ynmin)
                        Yz_train = (Yz_train-Yzmin)/(Yzmax-Yzmin)                    
                        Yz_test = (Yz_test-Yzmin)/(Yzmax-Yzmin)                        
                        Ydz_train = (Ydz_train-Ydzmin)/(Ydzmax-Ydzmin)
                        Ydz_test = (Ydz_test-Ydzmin)/(Ydzmax-Ydzmin)                    
                        kyokuritsu = keras.Input(shape=(1001,1), name="data")  
                        
                        vc=1e-3   # z
                        vcdz=1e-5 # delta z
                        vrb=1e-4  
                                                                                  
                        cz = layers.Conv1D(filters=32, kernel_size=15, strides=1, padding="same", activation="relu", kernel_regularizer=regularizers.l2(vc),input_shape=(1001, 1))(kyokuritsu)
                        cz = MaxPooling1D(5)(cz)
                        cz = layers.Conv1D(filters=32, kernel_size=15, strides=1, padding="same", activation="relu",kernel_regularizer=regularizers.l2(vc))(cz)
                        cz = layers.Conv1D(filters=32, kernel_size=15, strides=1, padding="same", activation="relu",kernel_regularizer=regularizers.l2(vc))(cz)
                        cz = MaxPooling1D(5)(cz)
                        cz = Dropout(0.5)(cz)
                        cz = layers.Conv1D(filters=32, kernel_size=15, strides=1, padding="same", activation="relu",kernel_regularizer=regularizers.l2(vc))(cz)
                        cz = layers.Conv1D(filters=32, kernel_size=15, strides=1, padding="same", activation="relu",kernel_regularizer=regularizers.l2(vc))(cz)
                        cz = MaxPooling1D(5)(cz)
                        cz = Dropout(0.5)(cz)
                        cz = layers.Conv1D(filters=32, kernel_size=15, strides=1, padding="same", activation="relu",kernel_regularizer=regularizers.l2(vc))(cz)
                        cz = MaxPooling1D()(cz)
                        cz = Dropout(0.5)(cz)
                        cz = Flatten()(cz)
                        cz = layers.Dense(128, activation="relu",kernel_regularizer=regularizers.l2(vc))(cz)
                        cz = layers.Dense(64, activation="relu",kernel_regularizer=regularizers.l2(vc))(cz)                        
                                          
                        ccdz = layers.Conv1D(filters=32, kernel_size=15, strides=1, padding="same", activation="relu", input_shape=(1001, 1),kernel_regularizer=regularizers.l2(vcdz))(kyokuritsu)
                        ccdz = MaxPooling1D(5)(ccdz)
                        ccdz = layers.Conv1D(filters=32, kernel_size=15, strides=1, padding="same", activation="relu",kernel_regularizer=regularizers.l2(vcdz))(ccdz)
                        ccdz = layers.Conv1D(filters=32, kernel_size=15, strides=1, padding="same", activation="relu",kernel_regularizer=regularizers.l2(vcdz))(ccdz)                        
                        ccdz = MaxPooling1D(5)(ccdz)
                        ccdz = Dropout(0.01)(ccdz)
                        ccdz = layers.Conv1D(filters=32, kernel_size=15, strides=1, padding="same", activation="relu",kernel_regularizer=regularizers.l2(vcdz))(ccdz)
                        ccdz = MaxPooling1D()(ccdz)
                        ccdz = Dropout(0.01)(ccdz)
                        ccdz = Flatten()(ccdz)
                        ccdz = layers.Dense(128, activation="relu",kernel_regularizer=regularizers.l2(vcdz))(ccdz)
                        ccdz = layers.Dense(64, activation="relu",kernel_regularizer=regularizers.l2(vcdz))(ccdz)
                                               

                        cc = layers.Conv1D(filters=32, kernel_size=15, strides=1, padding="same", activation="relu", input_shape=(1001, 1),kernel_regularizer=regularizers.l2(vcdz),bias_regularizer=regularizers.l2(vrb))(kyokuritsu)												
                        cc = MaxPooling1D(5)(cc)												
                        cc = layers.Conv1D(filters=32, kernel_size=15, strides=1, padding="same", activation="relu",kernel_regularizer=regularizers.l2(vcdz),bias_regularizer=regularizers.l2(vrb))(cc)												
                        cc = layers.Conv1D(filters=32, kernel_size=15, strides=1, padding="same", activation="relu",kernel_regularizer=regularizers.l2(vcdz),bias_regularizer=regularizers.l2(vrb))(cc)   												
                        cc = layers.Conv1D(filters=32, kernel_size=15, strides=1, padding="same", activation="relu",kernel_regularizer=regularizers.l2(vcdz),bias_regularizer=regularizers.l2(vrb))(cc)												
                        cc = layers.Conv1D(filters=32, kernel_size=15, strides=1, padding="same", activation="relu",kernel_regularizer=regularizers.l2(vcdz),bias_regularizer=regularizers.l2(vrb))(cc)												
                        cc = MaxPooling1D(5)(cc)												
                        cc = Dropout(0.01)(cc)																			
                        cc = layers.Conv1D(filters=32, kernel_size=15, strides=1, padding="same", activation="relu",kernel_regularizer=regularizers.l2(vcdz),bias_regularizer=regularizers.l2(vrb))(cc)												
                        cc = MaxPooling1D()(cc)												
                        cc = Dropout(0.01)(cc)												
                        cc = Flatten()(cc)												
                        cc = layers.Dense(128, activation="relu",kernel_regularizer=regularizers.l2(vcdz),bias_regularizer=regularizers.l2(vrb))(cc)												
                        cc = layers.Dense(64, activation="relu",kernel_regularizer=regularizers.l2(vcdz),bias_regularizer=regularizers.l2(vrb))(cc)    												


                        r = layers.Dense(1, activation="relu", name="r")(cc)    
                        n = layers.Dense(1, activation="relu", name="n")(cc)    
                        z = layers.Dense(1, activation="relu", name="z")(cz)     
                        deltaz = layers.Dense(1, activation="relu", name="deltaz")(ccdz)
                        
                    model = keras.Model(inputs=kyokuritsu, outputs=[r, n, z, deltaz], )
                                 
                    loss_weights=[10 ,20 ,10, 10]

                    model.compile(optimizer="adam",
                                  loss=["mse", "mse", "mse", "mse"],
                                  metrics=[["mae"], ["mae"], ["mae"], ["mae"]],
                                  loss_weights=loss_weights
                                  )
                    epoch=10000
                    batch_size=1024

                    checkpoint_callback = ModelCheckpoint(filepath='model_4lab_2.h5')                  
                    csv_logger = CSVLogger('training_4lab_2.csv', append=True)

                    model.summary()                    
                    history=model.fit(X_train, [Yr_train, Yn_train, Yz_train, Ydz_train], epochs=epoch, validation_split=0.3, batch_size=batch_size,callbacks=[checkpoint_callback, csv_logger],)                    
                    
                    
                    r_predtrain, n_predtrain, z_predtrain, deltaz_predtrain = model.predict(X_train)
                    r_predtest, n_predtest, z_predtest, deltaz_predtest = model.predict(X_test)
 
                    r_predtest = r_predtest*(Yrmax-Yrmin)+Yrmin
                    n_predtest = n_predtest*(Ynmax-Ynmin)+Ynmin
                    z_predtest = z_predtest*(Yzmax-Yzmin)+Yzmin                    
                    deltaz_predtest = deltaz_predtest*(Ydzmax-Ydzmin)+Ydzmin                   
                    
                    Yr_test = Yr_test*(Yrmax-Yrmin)+Yrmin
                    Yn_test = Yn_test*(Ynmax-Ynmin)+Ynmin
                    Yz_test = Yz_test*(Yzmax-Yzmin)+Yzmin
                    Ydz_test = Ydz_test*(Ydzmax-Ydzmin)+Ydzmin
                    

                    nch=len(r_predtest)

                    print("y1 MSE:%.4f" % mean_squared_error(Yr_test, r_predtest))
                    print("y2 MSE:%.4f" % mean_squared_error(Yn_test, n_predtest))
                    print("y3 MSE:%.4f" % mean_squared_error(Yz_test, z_predtest))
                    print("y4 MSE:%.4f\n" % mean_squared_error(Ydz_test*1000, deltaz_predtest*1000))                   
                    
                    print("y1 RMSE:%.4f" % mean_squared_error(Yr_test, r_predtest, squared=False))
                    print("y2 RMSE:%.4f" % mean_squared_error(Yn_test, n_predtest, squared=False))                   
                    print("y3 RMSE:%.4f" % mean_squared_error(Yz_test, z_predtest, squared=False))                 
                    print("y4 RMSE:%.4f" % mean_squared_error(Ydz_test*1000, deltaz_predtest*1000, squared=False))   

                    br=mean_squared_error(Yr_test, r_predtest, squared=False)
                    bn=mean_squared_error(Yn_test, n_predtest, squared=False)
                    bz=mean_squared_error(Yz_test, z_predtest, squared=False)
                    bdz=mean_squared_error(Ydz_test, deltaz_predtest, squared=False)
                    
                    mr=br/(1*(np.max(np.array(Yr_test))-np.min(np.array(Yr_test))))
                    mn=bn/(1*(np.max(np.array(Yn_test))-np.min(np.array(Yn_test))))
                    mz=bz/(1*(np.max(np.array(Yz_test))-np.min(np.array(Yz_test))))
                    mdz=bdz/(1*(np.max(np.array(Ydz_test))-np.min(np.array(Ydz_test))))
                    
                    print("y1 NRMSE:%.4f" % mr)
                    print("y2 NRMSE:%.4f" % mn)     
                    print("y3 NRMSE:%.4f" % mz)
                    print("y4 NRMSE:%.4f" % mdz)

                    for i in range(nch): 
                        with open('r_240716_4lab_1.txt', 'a') as f:                         
                            print("%.4f %.4f %.4f" %( Yr_test[i,0]*1000, r_predtest[i,0]*1000,(Yr_test[i,0]-r_predtest[i,0])*1000),file=f)
                        
                    for i in range(nch):   
                        with open('n_240716_4lab_1.txt', 'a') as f:  
                            print("%.4f %.4f %.4f" %(Yn_test[i,0], n_predtest[i,0],(Yn_test[i,0]-n_predtest[i,0])),file=f)                   

                    for i in range(nch):   
                        with open('z_240716_4lab_1.txt', 'a') as f:  
                            print("%.4f %.4f %.4f" %(Yz_test[i,0], z_predtest[i,0],(Yz_test[i,0]-z_predtest[i,0])),file=f)  
                                                 
                    print('Mean train : %.3f, test : %.3f' % (mean_absolute_error(Yr_train,r_predtrain), mean_absolute_error(Yr_test,r_predtest)))
                    print('Mean train : %.3f, test : %.3f' % (mean_absolute_error(Yn_train,n_predtrain), mean_absolute_error(Yn_test,n_predtest)))                  
                    print('Mean train : %.3f, test : %.3f' % (mean_absolute_error(Yz_train,z_predtrain), mean_absolute_error(Yz_test,z_predtest)))  

                    plt.figure(figsize=(7,5),facecolor="white")
                    plt.plot(history.history['r_loss'])
                    plt.plot(history.history['val_r_loss'])
                    plt.ylim([0,0.2])
                    plt.title('Model loss_r')
                    plt.ylabel('r_Loss(mse)')
                    plt.xlabel('Epoch')
                    plt.legend(['r_train', 'r_test'])
                    plt.show()                   

                                        # nlossを表示
                    plt.figure(figsize=(7,5),facecolor="white")
                    plt.plot(history.history['n_loss'])
                    plt.plot(history.history['val_n_loss'])
                    plt.ylim([0,0.2])
                    plt.title('Model loss_n')
                    plt.ylabel('n_Loss(mse)')
                    plt.xlabel('Epoch')
                    plt.legend(['n_train', 'n_test'])

                    plt.figure(figsize=(7,5),facecolor="white")
                    plt.plot(history.history['z_loss'])
                    plt.plot(history.history['val_z_loss'])
                    plt.ylim([0,0.2])
                    plt.title('Model loss_z')
                    plt.ylabel('z_Loss(mse)')
                    plt.xlabel('Epoch')
                    plt.legend(['z_train', 'z_test'])
                     #plt.savefig(os.path.join(save_dir,                    
                   
                    # deltazlossを表示
                    plt.figure(figsize=(7,5),facecolor="white")
                    plt.plot(history.history['deltaz_loss'])
                    plt.plot(history.history['val_deltaz_loss'])
                    plt.ylim([0,0.2])
                    plt.title('Model loss_deltaz')
                    plt.ylabel('deltaz_Loss(mse)')
                    plt.xlabel('Epoch')
                    plt.legend(['deltaz_train', 'deltaz_test'])

  