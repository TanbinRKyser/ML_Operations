# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 15:15:39 2025

@author: tanbi
"""

import pandas as pd
#%%
data = {'date': ['2023-01-01 14:23:45', '2023-01-02 08:15:00', '2023-01-03 22:45:30']}

df = pd.DataFrame(data)
df['date']=pd.to_datetime(df['date'])

#%%
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['hour'] = df['date'].dt.hour
df['day_name'] = df['date'].dt.day_name()
#%%
''' '''

data_2 = {'weather': ['Sunny', 'Clouds', 'Rainy'], 'temperature': [25.4, 22.8, 19.2]}
df = pd.DataFrame(data_2)

#df = df.astype(float)

non_numeric = df['weather'][ ~df['weather'].apply( 
                            lambda x: str( x ).replace( '.', '', 1 ).isdigit() 
                            ) 
                ]
print(non_numeric)

#%%
