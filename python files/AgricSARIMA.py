import pandas as pd
from statsmodels.tsa.stattools import adfuller
import numpy as np
import statsmodels.api as sm
from datetime import timedelta 


class AgricSARIMA():

    def __init__(self):
        self.data = None
    

    def adfuller_print_test(self, series , name='', signif = 0.05, verbose =False):
        adf_result = adfuller(series,autolag ='AIC')
        output = {'test statistic ': round(adf_result[0],4),'pvalue' :round(adf_result[1],4),'n_lags ':round(adf_result[2],4),'n_observations ':adf_result[3]}
        p_value=output['pvalue']
        
        if p_value<=signif:
            print("adf True")
        else:
            print("adf False")


    def sarima_compile(self,dataframe, list_of_incompatible_columns, number_of_years_to_predict):
        df_of_results = pd.DataFrame(columns = list_of_incompatible_columns, 
                                    index = pd.date_range( start = dataframe.index[-1]+timedelta(weeks=24) ,periods = 2*number_of_years_to_predict,freq='2BQ'))
        
        for i in list_of_incompatible_columns:
            data_series = dataframe[i]
            df_of_results[i] = self.sarima_agric(data_series , 2*number_of_years_to_predict)
            
        return df_of_results 

           
    def sarima_agric(self,data_series , time_steps):
        array_or_series_to_use = data_series - data_series.mean()
        negative_vals_indices  = array_or_series_to_use.index[array_or_series_to_use<0]
        data_scaled_log = np.log2(abs(array_or_series_to_use)) #cater for negative values,to avoid nan when we use np.log
        data_scaled_log[negative_vals_indices] = data_scaled_log[negative_vals_indices ]*-1          
        data_scaled  = data_scaled_log.diff().dropna() 
        
        self.adfuller_print_test(data_scaled)  # uncomment to check for stationarity 

        model= sm.tsa.statespace.SARIMAX(data_scaled, order=(1,1,1), seasonal_order=(1,1,2,6), verbose = False)
        model= model.fit(disp =False)
        initial_prediction = model.predict( start = len(data_series) , 
                                            end = len(data_series)+time_steps-1,
                                            dynamic = True 
                                          )
                                        
        #inverse transform the prediction
        inverse_diff_pred = data_scaled_log[-1]+initial_prediction.cumsum()# add differences to the last true value
        
        if all(h for h in inverse_diff_pred>0):
            prediction = 2**inverse_diff_pred
        else:
            prediction = 2**abs(inverse_diff_pred)
            for ind, item in enumerate(inverse_diff_pred):
                if item<0:
                    prediction[prediction.index[ind]]= prediction[prediction.index[ind]]*-1#cater for negative predictions
            
            prediction = prediction + data_series.mean()
        return prediction.values

    