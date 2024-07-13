import os
import pandas as pd
import numpy as np
import math
import datetime as dt
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error,mean_absolute_error,explained_variance_score,r2_score
from sklearn.metrics import mean_poisson_deviance,mean_gamma_deviance,accuracy_score
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from keras.src.models import Sequential
from keras.src.layers import Dense , Dropout
from keras.src.layers import LSTM

import matplotlib.pyplot as plt
from itertools import cycle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


maindf = pd.read_csv('BTC-USD.csv')

maindf['Date'] = pd.to_datetime(maindf['Date'],format='%Y-%m-%d')
y_2024 = maindf.loc[(maindf['Date'] >= '2024-01-01') & (maindf['Date'] < '2024-07-12')]
print(y_2024.drop(y_2024[['Adj Close','Volume']],axis=1))
monthvise = y_2024.groupby(y_2024['Date'].dt.strftime('%8'))[['Open','Close']].mean()
new_order = ['January','February','March','April','May','June','July']
monthvise = monthvise.reindex(new_order,axis=0)
print(monthvise)

fig = go.Figure()
fig.add_trace(go.Bar(
    x = monthvise.index,y=monthvise['Open'],name='Bitcoin Open Price',marker_color='crimson'
))
fig.add_trace(go.Bar(
    x=monthvise.index,y=monthvise['Close'],name='Bitcoin Close Price',marker_color='lightsalmon'

))
fig.update_layout(barmode='group',xaxis_tickangle=-45,title='Monthwise comparison between Bitcoin open and close price')
print(fig.show())

y_2024.groupby(y_2024['Date'].dt.strftime('%B'))['Low'].min()
monthvise_high = y_2024.groupby(maindf['Date'].dt.strftime('%B'))['High'].max()
monthvise_high = monthvise_high.reindex(new_order,axis=0)

monthvise_low = y_2024.groupby(y_2024['Date'].dt.strftime('%B'))['Low'].min()
monthvise_low = monthvise_low.reindex(new_order,axis=0)

fig = go.Figure()
fig.add_trace(go.Bar(
    x = monthvise_high.index,
    y = monthvise_high,
    name = 'Bitcoin High price',
    marker_color = 'rgb(0,153,204)'

))
fig.add_trace(go.Bar(
    x = monthvise_low.index ,
    y = monthvise_low ,
    name = 'Bitcoin low Price',
    marker_color = 'rgb(255,128,0)'
))
fig.update_layout(barmode='group',
                  title='Mouthwise High and Low Bitcoin price')
print(fig.show())

names = cycle(['Bitcoin Open price','Bitcoin Close Price','Bitcoin High Price','Bitcoin Low Price'])
fig = px.line(y_2024,x=y_2024.Date,y=[y_2024['Open'],y_2024['Close'],y_2024['High'],y_2024['Low']],
              labels={'Date':'Date','Value':'Bitcoin Value'})
fig.update_layout(title_text = 'Bitcoin Analysis Chart',font_size=15,font_color='black',legend_title_text='Bitcoin Parameters')
fig.for_each_trace(lambda t : t.update(name = next(names)))
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
print(fig.show())


