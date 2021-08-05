import pandas as pd
import numpy as np
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import grangercausalitytests
from datetime import timedelta 

class AgricVAR():
    def __init__(self, dataframe,number_of_years_to_predict):
        self.dataframe = dataframe
        self.number_of_years_to_predict = number_of_years_to_predict


    #Test for stationarity
    def adfuller_test(self, series , name='', signif = 0.05, verbose =False):
        adf_result = adfuller(series,autolag ='AIC')
        output = {  'test statistic ': round(adf_result[0],4),
                    'pvalue' :round(adf_result[1],4),'n_lags ':round(adf_result[2],4),
                    'n_observations ':adf_result[3]}

        p_value=output['pvalue']
        
        if p_value<=signif:
            return True 
                    
    def granger_column_combinations(self):
        if any(len(b)==1 for b in self.list_of_columns ):
            for i in self.list_of_columns:
                if len(i)==1:
                    dict_for_j = dict()
                    list_of_not_i = [x for x in self.list_of_columns if x!=i]
                    for k,j in enumerate(list_of_not_i):
                        direction_one = self.df.loc[i,j].values # i causing j
                        direction_two = self.df.loc[j,i].values # j causing i
                        flat_list = [item for sublist in direction_one for item in sublist]
                        flat_list2 = [item2 for sublist2 in direction_two for item2 in sublist2]
                        
                        try : 
                            if all(x < 0.1 for x in flat_list) and all(y < 0.1 for y in flat_list2): 
                                #0.051 to cater for 0.05   # *100 because the values it gave were too small
                                pair_the_flat_lists = [d for d in zip((np.array(flat_list))*100,(np.array(flat_list2))*100 )]
                                #get mean of std for it causing and being caused for that combination
                                mean_of_std = np.mean([np.std(b) for b in pair_the_flat_lists]) 

                                dict_for_j[k]=mean_of_std#pick the combination with least standard deviation  
                        except:
                            pass    

                    if bool(dict_for_j):#if it found combinations
                        min_val_in_dict = min(dict_for_j,key=dict_for_j.get)                  
                        list_of_not_i[min_val_in_dict].append(i[0])
                        #remove all columns that match to remain with those that don't
                        for column_name in list_of_not_i[min_val_in_dict]:
                            if column_name in self.non_compatible:
                                self.non_compatible.remove(column_name)
            
                        self.list_of_columns = list_of_not_i.copy()
 
            #self.granger_column_combinations()
        for incomp_column in self.non_compatible:
            self.list_of_columns.remove([incomp_column])
        
        return self.list_of_columns, self.non_compatible, self.df
        

    def granger_causality(self,variables,max_lag,test="ssr_chi2test",verbose=False):
        self.df = pd.DataFrame(np.zeros((len(variables),len(variables))), columns = variables, index=variables)
        
        for k in self.df.columns:
            for l in self.df.index:
                try:
                    test_result = grangercausalitytests(self.dataframe[[l,k]],maxlag=max_lag,verbose=False)
                    p_values = [round(test_result[i+1][0][test][1],4) for i in range(max_lag)]
                    if verbose : print(f"Y = {l},X={k}, P Values = {p_values}")
                    min_p_value=np.min(p_values)
                    self.df.loc[l,k]=min_p_value
                except:
                    self.df.loc[l,k] = "Didn't work"
        self.df.index = self.df.columns = [var for var in variables]
        self.non_compatible = [var for var in variables]
        # self.non_compatible list of all columns then remain with the ones with least causality
        self.list_of_columns =  [[var] for var in variables]
        #self.global_columns_list = self.list_of_columns.copy()

        return self.granger_column_combinations()


    

    def cointegration_test(self,list_of_combinations,alpha=0.05):
        for i in list_of_combinations:
            out = coint_johansen(self.dataframe[i],-1,1)
            d = {"0.90":0,"0.95":1,"0.99":2}
            traces=out.lr1
            cvts = out.cvt[:,d[str(1-alpha)]]
            
            for trace,cvt in zip(traces,cvts):
                if trace<alpha and cvt<alpha:
                    print("Null hypothesis True for ",i)#print if any implies that there is no cointegration i.e null hypothesis is true
            

    #make the df stationary #difference all columns same no. of times 
    def differencing_for_stationarity(self,dataframe, series_columns,count=0,diff_count_ = 0 ):
        #series columns = columns from the selected list of combinations
        for i in series_columns:
            if self.adfuller_test(dataframe[i],i):
                count=count+1
        
        if count!=len(series_columns):#difference if the number of stationary columns != total no. of columns i.e all columns must be stationary
            self.dataframe_diff= dataframe.diff().dropna().copy()
            self.no_of_times_diff = diff_count_+1
            cols = self.dataframe_diff.columns
            self.differencing_for_stationarity( self.dataframe_diff, cols, diff_count_ = self.no_of_times_diff )
        
        return self
    
    #Getting the optimal Lag
    def optimal_lag_and_residual_variance_check(self):
        model = VAR(self.dataframe_diff)
        
        if 'minimum_AIC' in locals():#remove the minimum AIC from any previous operations
            del minimum_AIC

        for i in range(1,12):
            result=model.fit(i)
            

            if 'minimum_AIC' not in locals():
                minimum_AIC = result.aic #creating the minimum aic
                optimal_lag_order = i

            if result.aic <= minimum_AIC:
                optimal_lag_order = i
                minimum_AIC = result.aic
                try: #cater for excess index
                    result=model.fit(i+1)
                    if result.aic > minimum_AIC:#stop when the AIC rises at any point after the first selection and pick the lag order before the rise
                        break;
                        
                except:
                    pass


        self.model_fitted = model.fit(optimal_lag_order)

        #Durbin Watson Statistic ::: Check for serial correlation , Ensure the model almost fully explains the variance
        out = durbin_watson(self.model_fitted.resid)
        for col,val  in zip(self.dataframe_diff.columns,out):
            if val>2+0.5 or val<2-0.5:  #the closer the value is to 2.0 the better 
                print(f"Durbin Watson (Residual) Error in {col} = {val}")
                
        return self


    #Invert the transformation
    def inverse_transform(self, original_df,df_forecast):
        df_fc = df_forecast.copy()
        columns=original_df.columns
        
        for col in columns:
            if self.no_of_times_diff >1:
                for diff_no in range(self.no_of_times_diff, 1,-1):
                    value_to_add = []
                    for i in range(diff_no, 1,-1):#repeat for all diferenced times
                        #inverse for any number of differenced times greater than 1  
                        value_for_subtraction = original_df[col].iloc[-(i-1)] - original_df[col].iloc[-i]
                        value_to_add.append(value_for_subtraction)
                        
                    for k in range(diff_no, 1,-1):#repeat for all diferenced times at this stage
                        for j in range(len(value_to_add)):
                            try:
                                value_to_add[j] = value_to_add[j+1]-value_to_add[j]
                            except:
                                pass 
                        if len(value_to_add)>1:
                            # select only values up to but not including the last one // each diff reduces length by 1
                            value_to_add = value_to_add[:-1].copy()
                
                    df_fc[col] = value_to_add[0]+df_fc[col].cumsum()
                                
            #roll_back last diff
            df_fc[col]=original_df[col].iloc[-1] + df_fc[col].cumsum()
        return df_fc
    


    def var_prediction(self,series_columns ):
        starionarity_df = self.dataframe[series_columns]
        self.differencing_for_stationarity(starionarity_df , series_columns )
        self.optimal_lag_and_residual_variance_check()
        #get_lag_order
        lag_order = self.model_fitted.k_ar
        forecast_input = self.dataframe_diff[-lag_order:]

        fc = self.model_fitted.forecast(y=forecast_input.values,steps=2*self.number_of_years_to_predict)
        
        df_forecast = pd.DataFrame(fc , index = pd.date_range(  start = self.dataframe_diff.index[-1]+timedelta(weeks=24),
                                                                periods = 2*self.number_of_years_to_predict,
                                                                freq='2BQ'),                                                     
                                    columns = self.dataframe_diff.columns) #set the resulting df to have proper index
        #timedelta is 24 weeks i.e 6 months 
        #add 6 months to the last index and continue from there
        
        return self.inverse_transform(starionarity_df , df_forecast)

    