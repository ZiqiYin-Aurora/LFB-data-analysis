'''
Author: Aurora Yin
Created: 27/10/2022
File description:
Useful tools for MSIN0143 group project.
'''

import os
import math
import shutil
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from pandas import Series, DataFrame 

# # Using Bokeh Visualisation 
import pandas_bokeh
pandas_bokeh.output_notebook() # show in notebook
# pandas_bokeh.output_file('bokeh plots.html') 
from bokeh.models import ColumnDataSource, Whisker
from bokeh.plotting import figure, show
from bokeh.sampledata.autompg2 import autompg2
from bokeh.transform import factor_cmap
from bokeh.layouts import gridplot

import numpy.linalg as la
from scipy import stats
from linearmodels import PooledOLS
import statsmodels.api as sm
from linearmodels import PanelOLS
from linearmodels import RandomEffects
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_white, het_breuschpagan


import shap
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import *


'''
Description:
Function to convert .xlsx file to .csv file

Argument(s):
fn - file name or path including file name

Return:
.csv filename

'''
def xlsx_to_csv_pd(fn):
    
    assert '.xlsx' in fn, "The file is not a .xlsx file!"
    
    data_xls = pd.read_excel(fn, index_col=0)
    csv_fn = fn.replace('.xlsx', '.csv') #replace excel document to csv file
    data_xls.to_csv(csv_fn, encoding='utf-8')
    
    return csv_fn


'''
Description:
Function to convert all .xlsx file in a folder into .csv file and store in a specific folder

Argument(s):
fp - folder path

Return:
By default

'''
def xlsx_to_csv_folder(fp1, fp2):
    
    for path, dns, fns in os.walk(fp1): # path--filepath  dns--dirnames  fns--filenames
        for fn in fns:
            if '.xls' in fn:
                data_xls = pd.read_excel(fp1 + '/' + fn, index_col=0)
                csv_fn = fn.replace(fn.split('.')[1], 'csv')
                if csv_fn not in fns:
                    data_xls.to_csv(fp2 + '/' + csv_fn, encoding='utf-8')
                
                # shutil.move(fp1 + '/' + fn,fp2) 
                print("Done with " + fn)
                
    print("Done!!")
    
    return 


'''
Description:
Get basic information of the file.

Argument(s):
csv_fn - filename

Return:
df - dataframe
'''
def get_data_info(csv_fn):
    df = pd.read_csv(csv_fn)
    print("Shape of Dataset:")
    print(df.shape)
    print("\nInfo of Dataset:")
    print(df.info())
    print("\nFirst 5 rows of Dataset:\n")
    print(df.head())
    
    return df


'''
Description:
Calculate the missing rate of each column in a dataframe.
Print out a list of percentages.

Argument(s):
df - dataframe

Return:
By default
'''
def missing_pct(df):
    print('Percentage of missing values:')

    for col in df.columns:
        pct_missing = np.mean(df[col].isnull()) #replace missing value with average values
        print('{} - {}%'.format(col, round(pct_missing*100)))
    return 


'''
Description:
Find the outliers of the dataframe.

Argument(s):
df - dataframe

Return:
res - a new dataframe which contains only the outliers' values with their row&column indexes.
'''
def outliers(df):
    outliers = []
    for year in df.columns.values:
        if str(year).isdigit():
            Q1 = df[year].quantile(0.25)
            Q3 = df[year].quantile(0.75)
            IQR = Q3 - Q1    #IQR is interquartile range. 

            filter = (df[year] >= Q1 - 1.5 * IQR) & (df[year] <= Q3 + 1.5 *IQR)
            s = df.loc[~filter][year]
            outliers.append(s)
    res = pd.concat(outliers, axis=1)
    return res
    
    
'''
Description:
Check if there is uninformative data in the dataframe.
And print corresponding information.

Argument(s):
df - dataframe

Return:
By default
'''
def uninformative(df): 
    num_rows = len(df.index)
    low_information_cols = []
    flag = False

    for col in df.columns: #remove uninformative data
        cnts = df[col].value_counts(dropna=False)
        top_pct = (cnts/num_rows).iloc[0]
        if top_pct > 0.90 and col in year_set:
            low_information_cols.append(col)
            print('{0}: {1:.5f}%'.format(col, top_pct*100))
            print(cnts)
            print()
            flag = True

    if flag == False:
        print("NO Uninformative data in this data set!")

        
'''
Description:
Check if there are duplicate data in the dataframe.
If exist, drop those duplicates.
And print corresponding information.

Argument(s):
df - dataframe

Return:
By default
'''
def duplicates(df):
    df_dedupped = df.drop_duplicates()
    if df.shape == df_dedupped.shape:
        print(f'NO duplicates data in this data set! Before: {df.shape} After: {df_dedupped.shape}')
    else:
        # Compare if there are duplicate rows
        print(f'HAVE duplicates data!!! Before: {df.shape} After: {df_dedupped.shape}')




'''
Description:
Check whether sequence seq contains ANY of the items in a set/list.

Argument(s):
seq - sequence

Return:
True/False
'''
def containsAny(seq, aset):
    return True if any(i in seq for i in aset) else False 


'''
Data Normalization
The final step for this dataset is to scale data values in range 1~100. 

Formula: zi = (xi – min(x)) / (max(x) – min(x)) * 100

where:
    zi: The ith normalized value in the dataset
    xi: The ith value in the dataset
    min(x): The minimum value in the dataset
    max(x): The maximum value in the dataset
'''
def scale(df):
    df_copy = df.copy()
    for cols, rows in df.iteritems():
        for data in df[cols]:
            index = df[df[cols]==data].index
            res = (data-df.min().min())/(df.max().max()-df.min().min())*100 #normalize data to 1-100
            df_copy.loc[index, cols] = round(res,2)
    return df_copy


'''
Word count function.
But the html-styled words are also calculated.
'''
import json
def wordcount(nb_filename):
    with open(nb_filename) as json_file:
        data = json.load(json_file)

    wordCount = 0 #calculate the number of word
    for each in data['cells']:
        cellType = each['cell_type']
        if cellType == "markdown":
            content = each['source']
            for line in content:
                temp = [word for word in line.split()]
                wordCount = wordCount + len(temp)
    return wordCount


'''
Description:
Test the optimal degree of polynomial regression model.
Will show a line graph of MSE vs degrees, for each feature.

Argument(s):
name - string, show in figure caption
x - string, feature(s) that want to be included in the polynomial model
y - string, dependent variable in the dataframe
df - dataframe

Return:
m_max - max degree
mse - list of MSE values in this test
'''
def optimal_degree(name, x, y, df):
    
    # Training set and test set = 8:2
    train_df = df[:int(len(df)*0.8)]
    test_df = df[int(len(df)*0.2):]

    train_x = train_df[x].values
    train_y = train_df[y].values

    test_x = test_df[x].values
    test_y = test_df[y].values

    train_x = train_x.reshape(len(train_x),1)
    test_x = test_x.reshape(len(test_x),1)
    train_y = train_y.reshape(len(train_y),1)

    mse = [] 
    m = 1 # init m
    m_max = 20 # set maximum degree
    while m <= m_max:
        model = make_pipeline(PolynomialFeatures(m, include_bias=False), LinearRegression())
        model.fit(train_x, train_y) # training model
        pre_y = model.predict(test_x) # test model
        mse.append(mean_squared_error(test_y, pre_y.flatten())) # calculate MSE
        m = m + 1

#     print("MSE result: ", mse)
    plt.figure(figsize=(5, 3))
    plt.plot([i for i in range(1, m_max + 1)], mse, 'r')
    plt.scatter([i for i in range(1, m_max + 1)], mse)
    plt.title(name+' - '+x)
    plt.xlabel("m")
    plt.ylabel("MSE")
    
    return m_max, mse


'''
Description:
Polynomial regression model.

Argument(s):
name - string, show in figure caption
x - string, feature(s) that want to be included in the polynomial model
y - string, dependent variable in the dataframe
df - dataframe
degree - maximum degree of X
subshow - True by defalt, the function won't show figures if False 

Return:
xx - a list of linspace, used to show the figure
results_2.flatten() - list of predicted results
'''
def poly_reg(name, x, y, df, degree, subshow=True):
    # 2009-2017 years len * 0.8
    train_df = df[:int(len(df)*0.8)]
    test_df = df[int(len(df)*0.2):]

    train_x = train_df[x].values
    train_y = train_df[y].values

    test_x = test_df[x].values
    test_y = test_df[y].values

    # Feature matrix
    poly_features_2 = PolynomialFeatures(degree=degree, include_bias=False)
    poly_train_x_2 = poly_features_2.fit_transform(train_x.reshape(len(train_x),1))
    poly_test_x_2 = poly_features_2.fit_transform(test_x.reshape(len(test_x),1))

    # Training and Predict
    model = LinearRegression()
    model.fit(poly_train_x_2, train_y.reshape(len(train_x),1)) # train

    xx = np.linspace(df[x].min()-1, df[x].max()+1, 100)
#     xx = np.linspace(0,100,100)
    results_2 = model.predict(poly_features_2.fit_transform(xx.reshape(len(xx), 1))) # predict
    results_2_test = model.predict(poly_test_x_2) # predict

    results_2.flatten() # predict result

    # print("Predict: ", results_2[-1])
    print("Absolute error: ", mean_absolute_error(test_y, results_2_test.flatten()))
    print("Squared error: ", mean_squared_error(test_y, results_2_test.flatten()))

    if subshow:
        # scatter plot  
        plt.figure(figsize=(3, 2))
        #plt.scatter(x,y,color = 'blue')
        years = list(set(i[1] for i in df_full.index.values))
        X = df[x]
        Y = df[y]
    #     plt.plot(results_2, )
    #     plt.plot(X, Y, color = 'blue', linewidth = 2)
        plt.scatter(X, Y)

        # predict curve
        plt.plot(xx, results_2, color = 'red', linewidth = 2)
    #     plt.plot(X.values, results_2, color = 'red', linewidth = 2)
        plt.title('Predict for '+ x +' in '+name)    
        plt.xlabel(x)    
        plt.ylabel(y)    
        plt.show()
    return xx, results_2.flatten()

'''

### Polynomial Regression for each feature ###
boroughs = sorted(list(set(i[0] for i in df_scaled.index.values)))
features = [i for i in df_scaled.columns if i not in ['London Incidents', 'Year']]
Y = 'London Incidents'

plots = []
for feature in features:    
    xx, yy = poly_reg('2009~2017', feature, Y, df_scaled, 4, subshow=False)
    
    p = figure(width=500, height=300)
    # add a line renderer
    p.scatter(df_scaled[feature], df_scaled[Y], size=5, alpha=0.8, color='orange')
    p.line(xx, yy, line_width=2, color='green')
    p.title.text = 'Incidents vs ' + feature
    p.title.text_font_size = "15px"
    plots.append(p)
    
grid = gridplot(plots, ncols=3, width=350, height=250)
show(grid)
  
'''