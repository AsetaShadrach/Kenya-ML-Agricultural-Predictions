import pandas as pd 
import numpy as np
from AgricLSTM import AgricLSTM

class AgricPreprocessing(AgricLSTM):

    def __init__(self,dataframe):
        self.dataframe = dataframe
        
    #Simple function to backfill for macrostatistics #Missing values for the first few years
    def backfill_data(self, index_several_steps_from_start):
        self.to_bfill = self.dataframe.loc[:index_several_steps_from_start,:]
        for i in self.to_bfill.columns:
            if self.to_bfill[i].count()<len(self.to_bfill):
                r_numbers = self.to_bfill[i].values[::-1]
                sum_of_rates=0
                rate = 0

                for j,k in enumerate(r_numbers): #Get mean rate of change and use it to back fill
                    if ~np.isnan(r_numbers[j+1]):
                        new_rate=abs(r_numbers[j+1]-k)/(k*10)
                        sum_of_rates = sum_of_rates+new_rate
                    else:
                        break;

                rate=sum_of_rates/(self.to_bfill[i].count()-1)
                for j,k in enumerate(r_numbers):
                    if np.isnan(k):
                        r_numbers[j]= float(format(r_numbers[j-1]*(1-rate), '.5f'))

                
                self.to_bfill[i][0:len(r_numbers)]=r_numbers[::-1] 
                self.dataframe.loc[:index_several_steps_from_start,:] = self.to_bfill.copy()
                
        return self.resampling_and_filling()


    def resampling_and_filling(self):
        self.dataframe.index = pd.date_range(start = pd.datetime(int(self.dataframe.index[0]),12,1),
                                                                periods=len(self.dataframe),freq='A')
        
        n = self.dataframe.resample('6M',convention='start',closed='right').last()
        
        for j in self.dataframe.columns:
            for i in range(0,len(n[j]),2):#fill up to where there are NANs 
                try:
                    if np.isnan(n[j][i])  and  np.isnan(n[j].iloc[i+1]):#if two empty values are 
                        #following each other then you are on the "corner" ; missing values
                        break;
                    else:
                        n[j][i+1] = n[j][i]
                        n[j][i] = n[j][i+1]
                except:#index error for 115
                    pass
                
        n = n.rolling(window=2).mean().dropna(how="all")#drop top rows with nans  
            
        return n

    def filling_the_nulls(self, data):        
        true_length = data.count()
        steps_val = len(data)-true_length
        vals_to_use = true_length- steps_val
        training_data = data[:true_length]
        
        return self.lstm_func(training_data, vals_to_use, true_length)


    def filling_df(self, index_several_steps_from_start):
        data_f = self.backfill_data(index_several_steps_from_start)
        df = data_f.copy()
        for i in df.columns:
            true_len = df[i].count()#check if there are any nulls
            if true_len < len(df[i]):
                pred=self.filling_the_nulls(df[i])
                data_f.loc[:-(len(pred)+1):-1, i] = pred
            
        return data_f

