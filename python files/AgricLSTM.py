import numpy as np
import pandas as pd
from datetime import timedelta 
from keras.models import Sequential
from keras.layers import LSTM,Dropout,Dense

class AgricLSTM():
    def __init__(self):
        self.data = None

    #LSTM function
    def model_for_lstm_filling(self,x_train ,y_train,lstm_units):
        model = Sequential()
        model.add(LSTM(units=lstm_units,return_sequences=True,input_shape=(x_train.shape[1],1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=lstm_units,return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=lstm_units))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))

        model.compile(optimizer="adam",loss="mean_squared_error")
        model.fit(x_train,y_train,epochs=10,batch_size=32,verbose=0)

        return model
    
    def lstm_func(self,array_or_series_to_use, vals_to_use, true_length ,lstm_units=30):
        #make the series stationary
        if all( h>0 for h in array_or_series_to_use):
            data_scaled_log = np.log2( array_or_series_to_use)            
            data_scaled  = data_scaled_log.diff().dropna()
            
        else:
            for index_of_zero_val,g in enumerate(array_or_series_to_use):
                if g == 0.0:
                    array_or_series_to_use[index_of_zero_val] = 1
                
            negative_vals_indices  = array_or_series_to_use.index[array_or_series_to_use<0]
            data_scaled_log = np.log2(abs(array_or_series_to_use)) #cater for negative values,to avoid nan whan we use np.log
            data_scaled_log[negative_vals_indices] = data_scaled_log[negative_vals_indices ]*-1          
            data_scaled  = data_scaled_log.diff().dropna()        

        #self.adfuller_print_test(data_scaled)  # uncomment to check for stationarity   
            
        x=[]
        y=[]
        for j in range(vals_to_use-1,true_length-1): # true_length-1 because of indexing
            x.append(data_scaled[j-(vals_to_use-1):j])
            y.append(data_scaled[j])
            

        x_train,y_train=np.array(x),np.array(y)
        x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
        model = self.model_for_lstm_filling(x_train, y_train,lstm_units)
        initial_prediction = model.predict(x_train)
        
        #inverse transform the prediction
        inverse_diff_pred = data_scaled_log[-1]+initial_prediction.cumsum() # add differences to the last true value
        if all(h for h in inverse_diff_pred>0):
            prediction = 2**inverse_diff_pred
        else:
            prediction = 2**abs(inverse_diff_pred)
            for ind, item in enumerate(initial_prediction):
                if item<0:
                    prediction[ind]= prediction[ind]*-1 # cater for negative predictions
        
        
        return prediction
    
    def lstm_compile(self,dataframe, list_of_incompatible_columns, number_of_years_to_predict):
        df_of_results = pd.DataFrame(columns = list_of_incompatible_columns, 
                                    index = pd.date_range( start = dataframe.index[-1]+timedelta(weeks=24) ,periods = 2*number_of_years_to_predict,freq='2BQ'))
        
        for i in list_of_incompatible_columns:
            data_series = dataframe[i]
            values_to_use = len(data_series) - 2*number_of_years_to_predict         
            df_of_results[i] = self.lstm_func(data_series , values_to_use ,len(data_series))
            
        return df_of_results 



    