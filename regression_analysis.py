# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# 
# %% [markdown]
# # Cardano Stake Pool Performance Analysis 🏴‍☠️
# 
# ## Overview
# 
# - We will be using various data science approaches to get a better understanding of Cardano stake pool performance using Python. We will use various linear regression models and statistical inference approaches to see what differences are there between performance of all pools, small vs. large, and what are the most important factors in predicting a stake pool's performance. This data science notebook will not go into the "rewards formula" of Cardano that many of us are familiar with, we will just be doing a basic analysis and explore the data we have available to us. 
# 

# %%
#Importing packages
import os
import math as m
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from scipy.optimize import curve_fit


pd.options.mode.chained_assignment = None  # default='warn'


# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)
pd.set_option('display.max_rows', 100)


# %%
# Load our data
pool_history_df = pd.read_csv('dataset.csv')


# %%
# take a quick peak into the dataset
pool_history_df


# %%
# Let's see what we are working with 
# This will show us some basic statistics about the data
pool_history_df.describe()


# %%
# Let's drop all NaN values from the df
pool_history_df = pool_history_df.dropna()

# Okay now let's make a column to convert the active_stake (in lovelace) to ada
pool_history_df['active_stake_ada'] = pool_history_df.iloc[:,4].values / 1000000
pool_history_df

# %% [markdown]
# # OLS regression
# Simple Single Variable Linear Regression

# %%
# We need to prepare the data for Liner Regression
x = pool_history_df.iloc[:,5].values

# Reshape the design matrix
x = x.reshape((-1, 1))
y = pool_history_df.iloc[:,3].values


# %%
print(x)
print(y)


# %%
model = LinearRegression().fit(x, y)


# %%
print('coefficient of determination:', model.score(x, y))
print('intercept:', model.intercept_)
print('slope:', model.coef_)


# %%
x_y = list(zip(pool_history_df.iloc[:,5].values,y))

# %% [markdown]
# ### Okay, so from the intial regression analysis we can see that our model of the roa based on active stake is quite poor. 

# %%
for i in range(0,len(x_y)):
    plt.scatter(x_y[i][0],x_y[i][1])


# %%
b1 = model.coef_
b0 = model.intercept_
abline_values = [b1 * i + b0 for i in x]
plt.scatter(x, y)
plt.plot(x, abline_values, 'b')
plt.title('fitted line y ='+str(b0)+' + '+str(b1)+'x')


# %%
y_pred = model.predict(x)
print('predicted response:', y_pred, sep='\n')


# %%
import plotly.express as px

fig = px.scatter(pool_history_df,
          x='active_stake_ada',
          y='roa',
          color='pool_id',
          size='active_stake_ada')

fig.update_yaxes(range=[0, 50])
fig.show()


# %%

ax = sns.heatmap(pool_history_df.corr(), annot = True,  cbar_kws= {'orientation': 'horizontal'} )
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)


# %%
epoch_273_df = pool_history_df.loc[pool_history_df['epoch'] == 273]

x_273 = epoch_273_df.iloc[:,5].values
x_273 = x_273.reshape((-1, 1))
y_273 = epoch_273_df.iloc[:,3].values
model_epoch_273 = LinearRegression().fit(x_273, y_273)
r_sq_273 = model_epoch_273.score(x_273,y_273)
print('coefficient of determination:', r_sq_273)
print('intercept:', model_epoch_273.intercept_)
print('slope:', model_epoch_273.coef_)


# %%
x_y_273= list(zip(epoch_273_df.iloc[:,5].values,y_273))
for i in range(0,len(x_y_273)):
    plt.scatter(x_y_273[i][0],x_y_273[i][1])


# %%
b1_273 = model_epoch_273.coef_
b0_273 = model_epoch_273.intercept_
abline_values = [b1_273 * i + b0_273 for i in x_273]
plt.scatter(x_273, y_273)
plt.plot(x_273, abline_values, 'b')
plt.title('fitted line y ='+str(b0_273)+' + '+str(b1_273)+'x')


# %%
fig = px.scatter(epoch_273_df,
          x='active_stake_ada',
          y='roa',
          color='pool_id',
          size='active_stake_ada',
          title='Epoch 273')

fig.update_yaxes(range=[0, 50])
fig.show()

# %% [markdown]
# # Logarithmic Regression

# %%
def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)


# %%
clean_df = pool_history_df.drop(['pool_id', 'pool_addr'], axis = 1)
clean_dataset(clean_df)


# %%
clean_df = clean_df.astype({"roa":'int', "active_stake_ada":'int'}) 
X = clean_df.loc[clean_df['active_stake_ada'] >= 1]
X = clean_df.iloc[:,3].values
Y = clean_df.iloc[:,1].values


# %%
X = X[X != 0]


# %%
if 0 in X:
	print("0 is in array X")
else:
	print("Clean")


# %%
import math

# Plot the data :
plt.scatter(X,Y)
plt.xlabel("active stake ada")
plt.ylabel("ROA")
plt.show()

# 1st column of our X matrix should be 1 :
n = len(X)
x_bias = np.ones((n,1))

print (X.shape)
print (x_bias.shape)

# Reshaping X :
X = np.reshape(X,(n,1))
print (X.shape)

# Going with the formula :
# Y = a + b*ln(X)
X_log = np.log(X)

# Append the X_log to X_bias :
x_new = np.append(x_bias,X_log,axis=1)

# Transpose of a matrix :
x_new_transpose = np.transpose(x_new)

# Matrix multiplication :
x_new_transpose_dot_x_new = x_new_transpose.dot(x_new)

# Find inverse :
temp_1 = np.linalg.inv(x_new_transpose_dot_x_new)

# Matrix Multiplication :
temp_2 = x_new_transpose.dot(Y)

# Find the coefficient values :
theta = temp_1.dot(temp_2)

# Plot the data :
a = theta[0]
b = theta[1]
Y_plot = a + b*np.log(X)
plt.scatter(X,Y)
plt.plot(X,Y_plot,c="r")

# Check the accuracy :
from sklearn.metrics import r2_score
Accuracy = r2_score(Y,Y_plot)
print (Accuracy)


# %%
clean_df.describe()

# %% [markdown]
# # Polynomial Regression
# %% [markdown]
# 
# %% [markdown]
# # Multiple Rergression 
# %% [markdown]
# # Advanced Linear Regression With statsmodels
# 
