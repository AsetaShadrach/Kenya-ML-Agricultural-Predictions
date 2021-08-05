from AgricPreprocess import AgricPreprocessing
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
 
#pesticide_data
pesticide_data=pd.read_csv("pesticide data.csv")
#fertilizer_data
fertilizer_data=pd.read_csv("fertilizer data.csv")


#Land Data 
land_data= pd.read_csv("land data.csv")
#Remove those with many zeros
land_data = land_data.set_index("Item").T.replace(0.0,np.nan)
land_data_to_be_retained = [i for i in land_data.columns if land_data[i].count()>int(len(land_data[i])*0.75) ]
land_data=land_data[land_data_to_be_retained].T
land_data.index.name = None
#Remove the constant Land Values... they don't seem to affect anything and bring an issue with calculation(VAR)
columns_to_remain = [i for i in land_data.index if land_data.T[i].nunique()>5]
land_data = land_data.T[columns_to_remain].T


# Crop Data
crop_data = pd.read_csv("crop data.csv")
crop_data = crop_data.set_index("Element") 
crop_data.index.name = None
#Add land variables to the crop data
crop_data = pd.concat([crop_data,land_data])


# Animal Data
animal_data  =  pd.read_csv("animal data.csv")

for  i in animal_data["Unnamed: 0"]:
    if i.startswith("("):
        item_index = animal_data["Unnamed: 0"].to_list().index(i)
        animal_data["Unnamed: 0"][item_index] = i.split("'")[1]+"["+i.split("'")[3]+"]"

               
animal_data = animal_data.set_index("Unnamed: 0")
animal_data.index.name = None
#animal_data


#Population Data
population_data = pd.read_csv("population data.csv", index_col="Element")
population_data.index.name=None


#Macro Statistics
macro_stats_data = pd.read_csv("macro statistics.csv")
macro_stats_data  = macro_stats_data.set_index("Item").T.replace(0.0,np.nan)
macro_stats_data_to_be_retained = [i for i in macro_stats_data.columns if macro_stats_data[i].count()>int(len(macro_stats_data[i])*0.75) ]
macro_stats_data = macro_stats_data[macro_stats_data_to_be_retained].T
macro_stats_data.index.name= None


full_crop_data = pd.concat([crop_data,population_data,macro_stats_data]).T
# Remove columns with same name since they contain similar data for this case
full_crop_data = full_crop_data.loc[:,~full_crop_data.columns.duplicated()]

full_animal_data = pd.concat([animal_data,population_data,macro_stats_data]).T
full_animal_data = full_animal_data.loc[:,~full_animal_data.columns.duplicated()]

#Filling the dataframes
full_crop_data_post = AgricPreprocessing(full_crop_data)
crop_full_df = full_crop_data_post.filling_df("1975") # year to back fill and lstm units

full_animal_data_post = AgricPreprocessing(full_animal_data)
animal_full_df = full_animal_data_post.filling_df("1975")


# save predicted/filled df to csv instead of rerunning the LSTM
animal_full_df.to_csv("animal_data_redone.csv")
crop_full_df.to_csv("crop_data_redone.csv")

#%%