#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 09:41:09 2021

@author: carolyndavis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools


from pydataset import data



# =============================================================================
# Intro to Anomally Detection: Messing Around with the Methodology 
# =============================================================================

# =============================================================================
# - Take a few minutes and jot down some example applications, specific problems, or
#  domains where detecting anomalies is super valuable and impactful.
# - Identify a few example applications, specific problems, or domains where the
#  outliers can skew the data in a negative way and those outliers may be safely
#  avoided for producing a model.
# 
# =============================================================================




# Ascombe's Quartet: Understanding that though descriptive data may appear similar 
#When viusalized depicts a different picture; the importance of visualizing data



url = "https://gist.githubusercontent.com/ryanorsinger/6218f5731f3df7cb4771ff3879fdeaa3/raw/88cb6bed276e2236c33df011bd753b6c73832c30/quartet.csv"

df = pd.read_csv(url)


df.head()

#output:
    
#    id dataset     x     y
# 0   0       I  10.0  8.04
# 1   1       I   8.0  6.95
# 2   2       I  13.0  7.58
# 3   3       I   9.0  8.81
# 4   4       I  11.0  8.33

list(itertools.product(['x','y'], ['50%', 'mean']))
#ouput: 
# [('x', '50%'), ('x', 'mean'), ('y', '50%'), ('y', 'mean')]

#-----------------------------------------------------------------

# check out the stats with a .describe()
df.groupby('dataset').describe()[list(itertools.product(['x','y'], ['50%', 'mean']))]
#output:
#            x          y          
#          50% mean   50%      mean
# dataset                          
# I        9.0  9.0  7.58  7.500909
# II       9.0  9.0  8.14  7.500909
# III      9.0  9.0  7.11  7.500000
# IV       8.0  9.0  7.04  7.500909

# print(type(stats))

sns.relplot(x='x', y='y', col='dataset', data=df)

# =============================================================================
# Playing around with the Swiss Data set
# =============================================================================
swiss_df = data('swiss')


#get a quick summary of the data:
swiss_df.describe()
#output:
    
#        Fertility  Agriculture  ...   Catholic  Infant.Mortality
# count  47.000000    47.000000  ...   47.00000         47.000000
# mean   70.142553    50.659574  ...   41.14383         19.942553
# std    12.491697    22.711218  ...   41.70485          2.912697
# min    35.000000     1.200000  ...    2.15000         10.800000
# 25%    64.700000    35.900000  ...    5.19500         18.150000
# 50%    70.400000    54.100000  ...   15.14000         20.000000
# 75%    78.450000    67.650000  ...   93.12500         21.700000
# max    92.500000    89.700000  ...  100.00000         26.600000

swiss_df.info()
#data is loolking clean 


# =============================================================================
# Visualizations:
# =============================================================================

#histographs:
    
    
# iterate through columns
for col in df.columns:
#     determine that it is a number type
    if np.issubdtype(df[col].dtype, np.number):
        df[col].hist()
        plt.title(col)
        plt.show()
        sns.boxplot(data=df, x=col)
        plt.show()




# =============================================================================
# Pairplot Visualizations
# =============================================================================
sns.pairplot(swiss_df)








# =============================================================================
 # =============================================================================
    
# steps to defining IQR/Tukey method:
# 1.) get the Q1 and Q3 values
# 2.)  determine our multiplier
# 3.) use these qualities to assert abnormalities



#zoooming into the examination column in the Swiss df 
swiss_df.Examination.quantile(0.25)
#output: 12.0


# start with an inner fence calculation
multiplier = 1.5
# calculate our q1 and q3
q1 = swiss_df.Examination.quantile(0.25)
q3 = swiss_df.Examination.quantile(0.75)
#q3= 22.0


iqr = q3 - q1
#ouput: iqr= 10.0

# inner or outer: 1.5 fence multiplier convention for inner, 3.0 mult convention for outer
# lower: q1 - mult* iqr
# upper: q3 + iqr*mult


inner_lower_fence = q1 - (1.5 * iqr)
inner_upper_fence = q3 + (3.0 * iqr)

#making an empty dataframe based off the examination column, with inputed outer and inners 
swiss_df[(swiss_df['Examination'] < inner_lower_fence) | (swiss_df['Examination'] > inner_upper_fence)]


# z-score:
# subtract the data point from the mean, divide by the standard deviation
# remember our z score calculation:
#  (x - x_mean) / x_std

#Switiching to zoom in on the infant mortality columns 

z_scores = (swiss_df['Infant.Mortality'] - swiss_df['Infant.Mortality'].mean()) / swiss_df['Infant.Mortality'].std()\

    
swiss_df['infant_zscore'] = z_scores


swiss_df['exam_zscores'] = z_scores


swiss_df[swiss_df['exam_zscores'].abs() >= 2]

#output:
#             Fertility  Agriculture  ...  infant_zscore  exam_zscores
# Porrentruy       76.1         35.3  ...       2.285664      2.285664
# La Vallee        54.3         15.2  ...      -3.138862     -3.138862

# [2 rows x 8 columns]

multiplier = 1.5
q1 = swiss_df['Infant.Mortality'].quantile(0.25)
q3 = swiss_df['Infant.Mortality'].quantile(0.75)

print(swiss_df.to_markdown())

#making another empty dataframe for the swiss dataset
swiss_df[(swiss_df['Infant.Mortality'] < inner_lower_fence) | (swiss_df['Infant.Mortality'] > inner_upper_fence)]
#output:
    
# Columns: [Fertility, Agriculture, Examination, Education, Catholic, Infant.Mortality, infant_zscore, exam_zscores]
# Index: []