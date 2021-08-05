import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class AgricPlots():
    def __init__ (self, dataframe):
        self.dataframe = dataframe

    def line_plotting(self, title):
        list_of_cols=[]
        for i in self.dataframe.max().sort_values(ascending=False).index:
            if len(list_of_cols) < 2:
                list_of_cols.append(i) 
                if np.array(self.dataframe[i].max()).std() < 0.10*self.dataframe[i].max():
                    pass
            else:  
                if self.dataframe[list_of_cols].max().std() < 0.10*self.dataframe[list_of_cols].max().max():
                    list_of_cols.append(i)
                else:
                    plt.figure()
                    plt.title(title,fontsize=25)
                    lines = plt.plot(self.dataframe[list_of_cols])
                    plt.legend(self.dataframe[list_of_cols].columns) 
                    list_of_cols=[]

    def correlation_plot(self,corr_col):  
        # cor_col --> column whose correlation you want to find with repect to the others  
        plt.figure(figsize=(20,5))
        cols_vs_dsq = self.dataframe.columns [self.dataframe.columns!=corr_col]
        plt.bar(cols_vs_dsq,self.dataframe.corr()[corr_col][cols_vs_dsq])
        plt.xticks(rotation = 90)
        plt.title(corr_col)
        plt.show()

    def regression_plots(self,target_column,number_of_plots_per_row):
        df_columns=self.dataframe.columns[self.dataframe.columns!=target_column]
        rows = int(np.ceil(len(df_columns)/number_of_plots_per_row))
        
        fig,axes=plt.subplots(rows,number_of_plots_per_row,sharey=True)
        for i in range (len(df_columns)):
            x=i//number_of_plots_per_row
            y=i%number_of_plots_per_row
            sns.regplot( x=self.dataframe[df_columns[i]], y=self.dataframe[target_column], ax=axes[x,y])