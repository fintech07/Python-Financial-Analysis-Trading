# Visualization

```python
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
```


```python
x = np.linspace(0, 5, 21)
y = x ** 2
```


```python
#Functional
plt.plot(x, y, 'r-')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
```


![png](output_2_0.png)



```python
plt.subplot(1, 2, 1)
plt.plot(x, y, 'r')
plt.subplot(1, 2, 2)
plt.plot(y, x, 'b')
plt.show()
```


![png](output_3_0.png)



```python
# oo
fig = plt.figure()
axes = fig.add_axes([0.1, 0.1, 1.2, 0.8])
axes.plot(x, y)
axes.set_xlabel('X Label')
axes.set_ylabel('Y Label')
axes.set_title('Title')
plt.show()
```


![png](output_4_0.png)



```python
fig = plt.figure()
axes1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
axes2 = fig.add_axes([0.2, 0.5, 0.4, 0.3])
axes1.plot(x, y)
axes1.set_title('Title')
axes2.plot(y, x)
axes2.set_title('Sub Title')
```




    Text(0.5, 1.0, 'Sub Title')




![png](output_5_1.png)



```python
fig, axes = plt.subplots(nrows = 1, ncols = 2)

for current_ax in axes:
    current_ax.plot(x, y)
    current_ax.set_title("title")
    
```


![png](output_6_0.png)



```python
axes[0].plot(x, y)
axes[0].set_title("First Plot")
axes[1].plot(y, x)
axes[1].set_title("Second Plot")
```




    Text(0.5, 1.0, 'Second Plot')



# Figure Size and DPI


```python
fig = plt.figure(figsize=(8, 2), dpi=200)
ax = fig.add_axes([0, 0, 1, 1])
ax.plot(x, y)
plt.show()
```


![png](output_9_0.png)



```python
fig = plt.figure(figsize=(8, 2), dpi=20)
ax = fig.add_axes([0, 0, 1, 1])
ax.plot(x, y)
plt.show()
```


![png](output_10_0.png)



```python
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 6))

axes[0][0].plot(x, y)
axes[1][0].plot(y, x)

axes[0][1].plot(x, x ** 2)
axes[0][1].plot(x, x ** 3)

axes[1][1].plot(x, x ** 0.5, label='X sqrt')
axes[1][1].plot(x, x ** 0.3, label='X Cubed')
axes[1][1].legend()

plt.tight_layout()

```


![png](output_11_0.png)



```python
fig.savefig('figure_test.png', dpi=200)
```


```python
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
ax.plot(x, y, color = '#FF8C00', linewidth=2, alpha=0.5, linestyle='--', 
        marker='o', markersize=20, markerfacecolor='yellow', markeredgecolor='red', markeredgewidth=3)

ax.set_xlim([0, 1])
ax.set_ylim([0, 2])
plt.show()
```


![png](output_13_0.png)



```python

```

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
 from datetime import datetime as dt
```


```python
my_year = 2017
my_month =1
my_day = 2
my_hour = 13
my_minute = 20
my_second = 15
```


```python
my_date = dt(2017, 1, 2, 0, 0)
```


```python
my_date_time = dt(my_year, my_month, my_day, my_hour, my_minute, my_second)
```


```python
my_date_time
```




    datetime.datetime(2017, 1, 2, 13, 20, 15)




```python
type(my_date)
```




    datetime.datetime




```python
type(my_date_time)
```




    datetime.datetime




```python
first_two = [dt(2016, 1, 1), dt(2016, 1, 2)]
```


```python
first_two
```




    [datetime.datetime(2016, 1, 1, 0, 0), datetime.datetime(2016, 1, 2, 0, 0)]




```python
type(first_two)
```




    list




```python
dt_ind = pd.DatetimeIndex(first_two)
```


```python
dt_ind
```




    DatetimeIndex(['2016-01-01', '2016-01-02'], dtype='datetime64[ns]', freq=None)




```python
data = np.random.randn(2, 2)
```


```python
data
```




    array([[ 1.17266737,  1.69298256],
           [-0.26544011,  1.26924552]])




```python
cols= ['a', 'b']
```


```python
df = pd.DataFrame(data, index=dt_ind, columns=cols)
```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2016-01-01</th>
      <td>1.172667</td>
      <td>1.692983</td>
    </tr>
    <tr>
      <th>2016-01-02</th>
      <td>-0.265440</td>
      <td>1.269246</td>
    </tr>
  </tbody>
</table>
</div>




```python
 df.index.argmax()
```




    1




```python
df.index.max()
```




    Timestamp('2016-01-02 00:00:00')



# Resampling


```python
df = pd.read_csv('silver_rates.csv', index_col='Date', parse_dates=True)
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>dAsk_open</th>
      <th>dAsk_high</th>
      <th>dAsk_low</th>
      <th>dAsk_close</th>
      <th>dBid_open</th>
      <th>dBid_high</th>
      <th>dBid_low</th>
      <th>dBid_close</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-08-20 14:53:01</th>
      <td>31.250120</td>
      <td>34.059719</td>
      <td>31.250120</td>
      <td>33.092663</td>
      <td>24.058703</td>
      <td>29.033866</td>
      <td>23.386813</td>
      <td>27.527786</td>
    </tr>
    <tr>
      <th>2020-08-20 14:54:01</th>
      <td>33.101373</td>
      <td>35.544810</td>
      <td>33.101373</td>
      <td>33.966106</td>
      <td>27.680442</td>
      <td>29.661571</td>
      <td>25.809777</td>
      <td>28.331556</td>
    </tr>
    <tr>
      <th>2020-08-20 14:55:01</th>
      <td>33.512670</td>
      <td>35.409447</td>
      <td>33.512670</td>
      <td>32.731117</td>
      <td>27.442044</td>
      <td>30.040872</td>
      <td>25.757473</td>
      <td>26.515337</td>
    </tr>
    <tr>
      <th>2020-08-20 14:56:01</th>
      <td>32.953497</td>
      <td>38.374924</td>
      <td>32.953497</td>
      <td>36.938822</td>
      <td>26.960129</td>
      <td>34.502927</td>
      <td>26.182525</td>
      <td>33.170154</td>
    </tr>
    <tr>
      <th>2020-08-20 14:57:01</th>
      <td>35.280327</td>
      <td>40.091153</td>
      <td>35.280327</td>
      <td>38.888639</td>
      <td>32.502927</td>
      <td>36.118225</td>
      <td>30.629955</td>
      <td>34.707211</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    DatetimeIndex: 22077 entries, 2020-08-20 14:53:01 to 2020-09-30 15:29:45
    Data columns (total 8 columns):
     #   Column      Non-Null Count  Dtype  
    ---  ------      --------------  -----  
     0   dAsk_open   22077 non-null  float64
     1   dAsk_high   22077 non-null  float64
     2   dAsk_low    22077 non-null  float64
     3   dAsk_close  22077 non-null  float64
     4   dBid_open   22077 non-null  float64
     5   dBid_high   22077 non-null  float64
     6   dBid_low    22077 non-null  float64
     7   dBid_close  22077 non-null  float64
    dtypes: float64(8)
    memory usage: 1.5 MB
    


```python
# df['Date'] = pd.to_datetime(df['Date'])
# df['Date'] = df['Date'].apply(pd.to_datetime)
```


```python
df.resample(rule='3H').mean()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>dAsk_open</th>
      <th>dAsk_high</th>
      <th>dAsk_low</th>
      <th>dAsk_close</th>
      <th>dBid_open</th>
      <th>dBid_high</th>
      <th>dBid_low</th>
      <th>dBid_close</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-08-20 12:00:00</th>
      <td>34.844231</td>
      <td>38.019255</td>
      <td>34.844231</td>
      <td>36.208967</td>
      <td>29.640929</td>
      <td>33.201499</td>
      <td>27.959599</td>
      <td>31.271007</td>
    </tr>
    <tr>
      <th>2020-08-20 15:00:00</th>
      <td>36.591096</td>
      <td>39.415138</td>
      <td>36.591096</td>
      <td>36.826618</td>
      <td>29.732016</td>
      <td>32.557460</td>
      <td>26.726296</td>
      <td>29.953440</td>
    </tr>
    <tr>
      <th>2020-08-20 18:00:00</th>
      <td>50.440275</td>
      <td>54.688804</td>
      <td>50.440275</td>
      <td>50.637591</td>
      <td>44.626024</td>
      <td>49.075697</td>
      <td>40.961414</td>
      <td>44.909409</td>
    </tr>
    <tr>
      <th>2020-08-20 21:00:00</th>
      <td>39.814060</td>
      <td>42.489044</td>
      <td>39.814060</td>
      <td>39.651191</td>
      <td>34.690150</td>
      <td>37.433994</td>
      <td>31.867674</td>
      <td>34.591218</td>
    </tr>
    <tr>
      <th>2020-08-21 00:00:00</th>
      <td>31.714012</td>
      <td>33.595156</td>
      <td>31.714012</td>
      <td>31.701509</td>
      <td>26.649041</td>
      <td>28.574963</td>
      <td>24.936129</td>
      <td>26.615225</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2020-09-30 03:00:00</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2020-09-30 06:00:00</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2020-09-30 09:00:00</th>
      <td>-226.948869</td>
      <td>-225.039741</td>
      <td>-228.831532</td>
      <td>-226.862388</td>
      <td>-232.273245</td>
      <td>-230.341255</td>
      <td>-234.071072</td>
      <td>-232.136208</td>
    </tr>
    <tr>
      <th>2020-09-30 12:00:00</th>
      <td>-207.564917</td>
      <td>-205.682523</td>
      <td>-209.240192</td>
      <td>-207.439653</td>
      <td>-213.603696</td>
      <td>-211.670179</td>
      <td>-215.224086</td>
      <td>-213.463863</td>
    </tr>
    <tr>
      <th>2020-09-30 15:00:00</th>
      <td>-195.460499</td>
      <td>-192.973781</td>
      <td>-197.720472</td>
      <td>-195.200084</td>
      <td>-201.136056</td>
      <td>-198.564879</td>
      <td>-203.239199</td>
      <td>-200.743992</td>
    </tr>
  </tbody>
</table>
<p>330 rows × 8 columns</p>
</div>




```python
def first_day(entry):
    if type(entry) is type(list):
        return entry
```


```python
df.resample(rule='3H').apply(first_day)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>dAsk_open</th>
      <th>dAsk_high</th>
      <th>dAsk_low</th>
      <th>dAsk_close</th>
      <th>dBid_open</th>
      <th>dBid_high</th>
      <th>dBid_low</th>
      <th>dBid_close</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-08-20 12:00:00</th>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>2020-08-20 15:00:00</th>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>2020-08-20 18:00:00</th>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>2020-08-20 21:00:00</th>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>2020-08-21 00:00:00</th>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2020-09-30 03:00:00</th>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>2020-09-30 06:00:00</th>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>2020-09-30 09:00:00</th>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>2020-09-30 12:00:00</th>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>2020-09-30 15:00:00</th>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
<p>330 rows × 8 columns</p>
</div>




```python
df['dBid_close'].resample('2D').mean().plot(kind='bar', figsize=(12, 9))
```


    <IPython.core.display.Javascript object>



<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAABLAAAAJYCAYAAABy5h8aAAAgAElEQVR4XuzdCbgsRXk/4AIXXBDcERQMKrhdFfdd0RgXMK5xSxQElUgQRWNYVASishpFRQW3QGKIiHFD4xbFLSSKogKCJsYAIqJEAhrIHxf4P9U359zL5eKt7ts1U1/Pe57HhwRqer55v5rqrt+Z6bPBVVdddVXyQ4AAAQIECBAgQIAAAQIECBAgQKBRgQ0EWI12RlkECBAgQIAAAQIECBAgQIAAAQKdgADLRCBAgAABAgQIECBAgAABAgQIEGhaQIDVdHsUR4AAAQIECBAgQIAAAQIECBAgIMAyBwgQIECAAAECBAgQIECAAAECBJoWEGA13R7FESBAgAABAgQIECBAgAABAgQICLDMAQIECBAgQIAAAQIECBAgQIAAgaYFBFhNt0dxBAgQIECAAAECBAgQIECAAAECAixzgAABAgQIECBAgAABAgQIECBAoGkBAVbT7VEcAQIECBAgQIAAAQIECBAgQICAAMscIECAAAECBAgQIECAAAECBAgQaFpAgNV0exRHgAABAgQIECBAgAABAgQIECAgwDIHCBAgQIAAAQIECBAgQIAAAQIEmhYQYDXdHsURIECAAAECBAgQIECAAAECBAgIsMwBAgQIECBAgAABAgQIECBAgACBpgUEWE23R3EECBAgQIAAAQIECBAgQIAAAQICLHOAAAECBAgQIECAAAECBAgQIECgaQEBVtPtURwBAgQIECBAgAABAgQIECBAgIAAyxwgQIAAAQIECBAgQIAAAQIECBBoWkCA1XR7FEeAAAECBAgQIECAAAECBAgQICDAMgcIECBAgAABAgQIECBAgAABAgSaFhBgNd0exREgQIAAAQIECBAgQIAAAQIECAiwzAECBAgQIECAAAECBAgQIECAAIGmBQRYTbdHcQQIECBAgAABAgQIECBAgAABAgIsc4AAAQIECBAgQIAAAQIECBAgQKBpAQFW0+1RHAECBAgQIECAAAECBAgQIECAgADLHCBAgAABAgQIECBAgAABAgQIEGhaQIDVdHsUR4AAAQIECBAgQIAAAQIECBAgIMAyBwgQIECAAAECBAgQIECAAAECBJoWEGA13R7FESBAgAABAgQIECBAgAABAgQICLDMAQIECBAgQIAAAQIECBAgQIAAgaYFBFhNt0dxBAgQIECAAAECBAgQIECAAAECAixzgAABAgQIECBAgAABAgQIECBAoGkBAVbT7VEcAQIECBAgQIAAAQIECBAgQICAAMscIECAAAECBAgQIECAAAECBAgQaFpAgNV0exRHgAABAgQIECBAgAABAgQIECAgwDIHCBAgQIAAAQIECBAgQIAAAQIEmhYQYDXdHsURIECAAAECBAgQIECAAAECBAgIsMwBAgQIECBAgAABAgQIECBAgACBpgUEWE23R3EECBAgQIAAAQIECBAgQIAAAQICLHOAAAECBAgQIECAAAECBAgQIECgaQEBVtPtURwBAgQIECBAgAABAgQIECBAgIAAyxwgQIAAAQIECBAgQIAAAQIECBBoWkCA1XR7FEeAAAECBAgQIECAAAECBAgQICDAMgcIECBAgAABAgQIECBAgAABAgSaFhBgNd0exREgQIAAAQIECBAgQIAAAQIECAiwzAECBAgQIECAAAECBAgQIECAAIGmBQRYTbdHcQQIECBAgAABAgQIECBAgAABAgIsc4AAAQIECBAgQIAAAQIECBAgQKBpAQFW0+1RHAECBAgQIECAAAECBAgQIECAgADLHCBAgAABAgQIECBAgAABAgQIEGhaQIDVdHsUR4AAAQIECBAgQIAAAQIECBAgIMAyBwgQIECAAAECBAgQIECAAAECBJoWEGA13R7FESBAgAABAgQIECBAgAABAgQICLDMAQIECBAgQIAAAQIECBAgQIAAgaYFBFhNt0dxBAgQIECAAAECBAgQIECAAAECAixzgAABAgQIECBAgAABAgQIECBAoGkBAVbT7VEcAQIECBAgQIAAAQIECBAgQICAAMscIECAAAECBAgQIECAAAECBAgQaFpAgNV0exRHgAABAgQIECBAgAABAgQIECAgwDIHCBAgQIAAAQIECBAgQIAAAQIEmhYQYDXdHsURIECAAAECBAgQIECAAAECBAgIsMwBAgQIECBAgAABAgQIECBAgACBpgUEWE23R3EECBAgQIAAAQIECBAgQIAAAQICLHOAAAECBAgQIECAAAECBAgQIECgaQEBVtPtURwBAgQIECBAgAABAgQIECBAgIAAyxwgQIAAAQIECBAgQIAAAQIECBBoWkCA1XR7FEeAAAECBAgQIECAAAECBAgQICDAMgcIECBAgAABAgQIECBAgAABAgSaFhBgNd0exREgQIAAAQIECBAgQIAAAQIECAiwzAECBAgQIECAAAECBAgQIECAAIGmBQRYTbenX3Ff/vKX0xFHHJG++c1vpp/85CfpIx/5SHrKU56yfJCrrroqHXDAAend7353uuSSS9JDH/rQ9M53vjNts802y2MuvvjitOeee6aTTjopbbjhhunpT396estb3pI23njj4mKuvPLKdMEFF6Sb3OQmaYMNNih+nIEECBAgQIAAAQIECBAgQGCIQN7v/vKXv0xbbLFFt5f1Mz0BAdaEevqpT30q/fM//3O6733vm572tKddI8A67LDD0iGHHJKOO+64tPXWW6f9998/nXHGGemss85KN7jBDTqJJzzhCV34dcwxx6Rf//rXaZdddkn3v//90/HHH18sdf7556ctt9yyeLyBBAgQIECAAAECBAgQIEBgDIEf/ehH6Xa3u90Yh3KMxgQEWI01ZKxy8iefVv8EVk6jcxL953/+5+mVr3xl9zSXXnpp2myzzdKxxx6bnv3sZ6ezzz473e1ud0unnnpqut/97teN+fSnP5122GGHlEOp/PiSn3zcm970pikvHJtssknJQ4whQIAAAQIECBAgQIAAAQKDBX7xi190H6TI3zbadNNNBx/HA9sVEGC125v1qmzNAOuHP/xhuuMd75i+9a1vpe2222752I985CO7/z9/TfB973tfF3D993//9/J//81vftN9OuvEE09MT33qU4tqygtHXjBykCXAKiIziAABAgQIECBAgAABAgTWQ8A+dD3wgjxUgBWkUX3LXDPAOuWUU7p7XuV7U22++ebLh3vmM5/Z3afqhBNOSAcffHD39cLvf//7V3u6W9/61umggw5Ku++++1rLuOKKK1L+39LPUvItwOrbNeMJECBAgAABAgQIECBAYIiAAGuIWqzHCLBi9au42lkGWAceeGAXcK35I8AqbpeBBAgQIECAAAECBAgQILAeAgKs9cAL8lABVpBG9S1zll8h9Amsvt0xngABAgQIECBAgAABAgTGFBBgjanZ5rEEWG32Zb2rurabuOcbuOf7XOWf/AbPXw9c8ybu3/jGN7q/ZJh/PvvZz6bHP/7xvW7ibuFY7/Y5AAECBAgQIECAAAECBAj0ELAP7YEVdKgAK2jj1lb2//zP/6Qf/OAH3X+6973vnd70pjelRz3qUenmN7952mqrrdJhhx2WDj300O4+V1tvvXXaf//90+mnn57OOuus7kbt+ecJT3hC+ulPf5qOPvro9Otf/zrtsssu3V8kPP7444ulLBzFVAYSIECAAAECBAgQIECAwAgC9qEjIDZ+CAFW4w3qU94Xv/jFLrBa82fnnXfuPmV11VVXpQMOOCC9613v6v606MMe9rD0jne8I2277bbLD7n44ovTS17yknTSSSelDTfcMD396U9Pb33rW9PGG29cXIqFo5jKQAIECBAgQIAAAQIECBAYQcA+dATExg8hwGq8QRHLs3BE7JqaCRAgQIAAAQIECBAgEFfAPjRu70orF2CVShlXLGDhKKYykAABAgQIECBAgAABAgRGELAPHQGx8UMIsBpvUMTyLBwRu6ZmAgQIECBAgAABAgQIxBWwD43bu9LKBVilUsYVC1g4iqkMJECAAAECBAgQIECAAIERBOxDR0Bs/BACrMYbFLE8C0fErqmZAAECBAgQIECAAAECcQXsQ+P2rrRyAVaplHHFAhaOYioDCRAgQIAAAQIECBAgQGAEAfvQERAbP4QAq/EGRSzPwhGxa2omQIAAAQIECBAgQIBAXAH70Li9K61cgFUqZVyxgIWjmMpAAgQIECBAgAABAgQIEBhBwD50BMTGDyHAarxBEcuzcETsmpoJECBAgAABAgQIECAQV8A+NG7vSisXYJVKGVcsYOEopjKQAAECBAgQIECAAAECBEYQsA8dAbHxQwiwGm9QxPIsHBG7pmYCBAgQIECAAAECBAjEFbAPjdu70soFWKVSxhULWDiKqQwkQIAAAQIECBAgQIAAgREE7ENHQGz8EAKsxhsUsTwLR8SuqZkAAQIECBAgQIAAAQJxBexD4/autHIBVqmUccUCFo5iKgMJECBAgAABAgQIECBAYAQB+9AREBs/hACr8QZFLM/CEbFraiZAgAABAgQIECBAgEBcAfvQuL0rrVyAVSplXLGAhaOYykACBAgQIECAAAECBAgQGEHAPnQExMYPIcBqvEERy7NwROyamgkQIECAAAECBAgQIBBXwD40bu9KKxdglUoZVyxg4SimMpAAAQIECBAgQIAAAQIERhCwDx0BsfFDCLAab1DE8iwcEbumZgIECBAgQIAAAQIECMQVsA+N27vSygVYpVLGFQtYOIqpDCRAgAABAgQIECBAgACBEQTsQ0dAbPwQAqzG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0ylk6gtHrYVZgDWd94BXQoAAAQIECBDoK1DrGlOAJcBacy5OdU5MfR/ad02Z4ngB1hS7OufXNPWFo9bFhQBrzhPX0xMgQIAAAQIE5ihQ6xpzqmFF31ZF882vL1rN86536vvQvnN+iuMFWFPs6pxf09QXjloLswBrzhPX0xMgQIAAAQIE5ihQ6xpTgLWyqdF8I9Y8b+Op70PnuDw189QCrGZaMZ1Cpr5w1FqYBVjTeQ94JQQIECBAgACBvgK1rjEFWAKsNefiVOfE1PehfdeUKY4XYE2xq3N+TVNfOGpdXAiw5jxxPT0BAgQIECBAYI4Cta4xpxpW9G1VNN/8+qLVPO96p74P7TvnpzhegDXFrs75NU194ai1MAuw5jxxPT0BAgQIECBAYI4Cta4xBVgrmxrNN2LN8zae+j50jstTM08twGqmFdMpZOoLR62FWYA1nfeAV0KAAAECBAgQmLpArWtigduqmcN4pUXpnJj6PnTqa0rJ6xNglSgZ00tg6gtHrRNJn8W5V0OC/sap72s0ngABAgQIECBAYHYCta6JS8OKvq80Wr359UWred71Tn0f2nfOT3G8AGuKXZ3za5r6wlFrYRZgzXnienoCBAgQIECAAIFigVrXxAKsVS1gvNKidE5MfR9a/Oac8EAB1oSbO6+XNvWFo9aJpM/i3Le3tWouPZn0rdd4AgQIECBAgACBtgWiXV9Gqzd3P1rN86536vvQtleE2VQnwJqN80ye5Q1veEP65Cc/mb797W+n61//+umSSy65xvOed955affdd08nn3xy2njjjdPOO++cDjnkkHTd6153eewXv/jF9IpXvCJ997vfTVtuuWV6zWtek57//OcXv4apLxy1FmYBVvEUM5AAAQIECBAgQGDOArWuiWv9gjRavQKsVRO8dE5MfR8657d8E08vwGqiDeMUccABB6Sb3vSm6fzzz0/vfe97rxFg/fa3v03bbbddus1tbpOOOOKI9JOf/CTttNNO6UUvelE6+OCDuyL+8z//M61YsSK9+MUvTi984QvT5z//+bTXXnt1wdjjHve4okKnvnDUOvkJsIqml0EECBAgQIAAAQINCNS6Ji4NK/oSRKtXgCXA6jvHF2G8AGuCXT722GO70GnNT2B96lOfSk984hPTBRdckDbbbLPulR999NFpn332SRdddFH3qa38f+ew6swzz1yWefazn90d69Of/nSRlgCriGmtg5ywh9t5JAECBAgQIECAwOwEogVC0eoVYAmwZvdujvNMAqw4vSqu9NoCrNe+9rXp4x//ePcVw6Wf/ImrO9zhDum0005L9773vdMjHvGIdJ/73CcdeeSRy2P++q//ugvELr300rXWcMUVV6T8v6WfHGDlrx7m8Ztssklx3VEG1jr55dcvwIoyC9RJgAABAgQIEFhsgVrXxK6HV80rxistSufE1D9IsdgrzspXL8Ca4Cy4tgBrt912S+eee276zGc+s/yqL7/88nTjG984/eM//mN6whOekLbddtu0yy67pP322295TP5vO+64Y8pjb3jDG15D7MADD0wHHXTQNf69AKv/5CpdnPseOdrJr+/rM54AAQIECBAgQGC2AtGuL6PVm7sZreZ51yvAmu0aMI9nE2DNQ73Hc+67777psMMO+52POPvss9Nd7nKX5TGzDrB8AqtHQ9cxVIA1nqUjESBAgAABAgQI1BOYd1jR95VFq1eAtarDpXskAVbfd0W88QKsxnuW703185///HdWmb8CmO9ftfQz668Qrlnc1BeOWie/7Fi6OPedtrVqrlVv39dnPAECBAgQIECAwGwFol1fRqtXgCXAmu07OsazCbBi9KlXleu6iXv+64O3vvWtu2O+613vSn/xF3+Rfvazn6WNNtqou4l7/srgGWecsfycf/zHf5wuvvhiN3H/P5FaJz8BVq9pbjABAgQIECBAgMAcBWpdE9f6BWm0egVYAqw5vr2bfWoBVrOt6V/Yeeed1wVN+UbtRxxxRPrKV77SHeROd7pT2njjjdNvf/vbtN1226UtttgiHX744enCCy9Mz3ve89ILX/jCdPDBB3dj803dV6xYkfbYY4+06667pi984QvppS99afeXCR/3uMcVFeUTWEVMax3khD3cziMJECBAgAABAgRmJxAtEIpWrwBLgDW7d3OcZxJgxenVOit9/vOfn4477rhrjDv55JPT9ttv3/37fBP33XffPX3xi1/sbt6+8847p0MPPTRd97rXXX5c/m8vf/nL01lnnZVud7vbpf333z/lY5f+CLBKpa45ToA13M4jCRAgQIAAAQIEZicQLRCKVq8AS4A1u3dznGcSYMXpVZhKBVjDWyXAGm7nkQQIECBAgAABArMTiBYIRatXgCXAmt27Oc4zCbDi9CpMpQKs4a0SYA2380gCBAgQIECAAIHZCUQLhKLVK8ASYM3u3RznmQRYcXoVplIB1vBWCbCG23kkAQIECBAgQIDA7ASiBULR6hVgCbBm926O80wCrDi9ClOpAGt4qwRYw+08kgABAgQIECBAYHYC0QKhaPUKsARYs3s3x3kmAVacXoWpVIA1vFUCrOF2HkmAAAECBAgQIDA7gWiBULR6BVgCrNm9m+M8kwArTq/CVCrAGt4qAdZwO48kQIAAAQIECBCYnUC0QChavQIsAdbs3s1xnkmAFadXYSoVYA1vlQBruJ1HEiBAgAABAgQIzE4gWiAUrV4BlgBrdu/mOM8kwIrTqzCVCrCGt0qANdzOIwkQIECAAAECBGYnEC0QilavAEuANbt3c5xnEmDF6VWYSgVYw1slwBpu55EECBAgQIAAAQKzE4gWCEWrV4AlwJrduznOMwmw4vQqTKUCrOGtEmANt/NIAgQIECBAgACB2QlEC4Si1SvAEmDN7t0c55kEWHF6FaZSAdbwVgmwhtt5JAECBAgQIECAwOwEogVC0eoVYAmwZvdujvNMAqw4vQpTqQBreKsEWMPtPJIAAQIECBAgQGB2AtECoWj1CrAEWLN7N8d5JgFWnF6FqVSANbxVAqzhdh5JgAABAgQIECAwO4FogVC0egVYAqzZvZvjPJMAK06vwlQqwBreKgHWcDuPJECAAAECBAgQmJ1AtEAoWr0CLAHW7N7NcZ5JgBWnV2EqFWANb5UAa7idRxIgQIAAAQIECMxOIFogFK1eAZYAa3bv5jjPJMCK06swlQqwhrdKgDXcziMJECBAgAABAgRmJxAtEIpWrwBLgDW7d3OcZxJgxelVmEoFWMNbJcAabueRBAgQIECAAAECsxOIFghFq1eAJcCa3bs5zjMJsOL0KkylAqzhrRJgDbfzSAIECBAgQIAAgdkJRAuEotUrwBJgze7dHOeZBFhxehWmUgHW8FYJsIbbeSQBAgQIECBAgMDsBKIFQtHqFWAJsGb3bo7zTAKsOL0KU6kAa3irBFjD7TySAAECBAgQIEBgdgLRAqFo9QqwBFizezfHeSYBVpxehalUgDW8VbUCrOEVeSQBAgQIECBAgACBawpEC4Si1SvAEmBZd64pIMAyK0YXEGANJxVgDbfzSAIECBAgQIAAgdkJRAuEotUrwBJgze7dHOeZBFhxehWmUgHW8FYJsIbbeSQBAgQIECBAgMDsBKIFQtHqFWAJsGb3bo7zTAKsOL0KU6kAa3irBFjD7TySAAECBAgQIEBgdgLRAqFo9QqwBFizezfHeSYBVpxehalUgDW8VQKs4XYeSYAAAQIECBAgMDuBaIFQtHoFWAKs2b2b4zyTACtOr8JUKsAa3ioB1nA7jyRAgAABAgQIEJidQLRAKFq9AiwB1uzezXGeSYAVp1dhKhVgDW+VAGu4nUcSIECAAAECBAjMTiBaIBStXgGWAGt27+Y4zyTAitOrMJUKsIa3SoA13M4jCRAgQIAAAQIEZicQLRCKVq8AS4A1u3dznGcSYMXpVZhKBVjDWyXAGm7nkQQIECBAgAABArMTiBYIRatXgCXAmt27Oc4zCbDi9CpMpQKs4a0SYA2380gCBAgQIECAAIHZCUQLhKLVK8ASYM3u3RznmQRYcXoVplIB1vBWCbCG23kkAQIECBAgQIDA7ASiBULR6hVgCbBm926O80wCrDi9ClOpAGt4qwRYw+08kgABAgQIECBAYHYC0QKhaPUKsARYs3s3x3kmAVacXoWpVIA1vFUCrOF2HkmAAAECBAgQIDA7gWiBULR6BVgCrNm9m+M8kwArTq/CVCrAGt4qAdZwO48kQIAAAQIECBAgcG0CAqz+gVDf2TRv46nvQ/v2Y4rjBVhT7OqcX9PUF45aC3NumwBrzpPX0xMgQIAAAQIECExSoNY1fM3r92g1z7veqe9DJ/nG7PmiBFg9wQxft8DUF45aC7MAa91zywgCBAgQIECAAAECiyJQa99RK3Sbd71T34cuyrz/Xa9TgGUWjC4w9YWj1sIswBp9KjogAQIECBAgQIAAgbACtfYdAqywU2LhCxdgLfwUGB9AgDXctNbJZHhFHkmAAAECBAgQIECAwDwEBFgr1Uv3SFPfh85jDrb2nAKs1joygXqmvnDUOpH0WZwnME28BAIECBAgQIAAAQIEfodArX1HaSDUtznzrnfq+9C+/ZjieAHWFLs659c09YWj1sIswJrzxPX0BAgQIECAAAECBAgMFqi1TyoN3Ka+Dx3cmAk9UIA1oWa28lKmvnDUWpgFWK3MYHUQIECAAAECBAgQINBXoNY+SYDVtxPTHS/Amm5v5/bKBFjD6UsX5+HP4JEECBAgQIAAAQIECBAYX0CANb6pI15dQIBlRowuIMAaTirAGm7nkQQIECBAgAABAgQIzE9AgDU/+0V5ZgHWonR6hq9TgDUcW4A13M4jCRAgQIAAAQIECBCYn4AAa372i/LMAqxF6fQMX6cAazi2AGu4nUcSIECAAAECBAgQIDA/AQHW/OwX5ZkFWIvS6Rm+TgHWcGwB1nA7jyRAgAABAgQIECBAYH4CAqz52S/KMwuwFqXTM3ydAqzh2AKs4XYeSYAAAQIECBAgQIDA/AQEWPOzX5RnFmAtSqdn+DoFWMOxBVjD7TySAAECBAgQIECAAIH5CQiw5me/KM8swFqUTs/wdQqwhmMLsIbbeSQBAgQIECBAgAABAvMTEGDNz35RnlmAtSidnuHrFGANxxZgDbfzSAIECBAgQIAAAQIE5icgwJqf/aI8swBrUTo9w9cpwBqOLcAabueRBAgQIECAAAECBAjMT0CANT/7RXlmAdaidHqGr1OANRxbgDXcziMJECBAgAABAgQIEJifgABrfvaL8swCrEXp9AxfpwBrOLYAa7idRxIgQIAAAQIECBAgMD8BAdb87BflmQVYi9LpGb5OAdZwbAHWcDuPJECAAAECBAgQIEBgfgICrPnZL8ozC7AWpdMzfJ0CrOHYAqzhdh5JgAABAgQIECBAgMD8BARY87NflGcWYC1Kp2f4OgVYw7EFWMPtPJIAAQIECBAgQIAAgfkJCLDmZ78ozyzAWpROz/B1CrCGYwuwhtt5JAECBAgQIECAAAEC8xMQYM3PflGeWYC1KJ2e4esUYA3HFmANt/NIAgQIECBAgAABAgTmJyDAmp/9ojyzAGtROj3D1ynAGo4twBpu55EECBAgQIAAAQIECMxPQIA1P/tFeWYB1qJ0eoavU4A1HFuANdzOIwkQIECAAAECBAgQmJ+AAGt+9ovyzAKsRen0DF+nAGs4tgBruJ1HEiBAgAABAgQIECAwPwEB1vzsF+WZBViL0ukZvs6pB1gzpPRUBAgQIECAAAECBAgQCCEgwArRptBFCrBCt6/N4gVYbfZFVQQIECBAgAABAgQIEKglIMCqJeu4SwICLHNhdAEB1uikDkiAAAECBAgQIECAAIGmBQRYTbdnEsUJsCbRxrZehACrrX6ohgABAgQIECBAgAABArUFBFi1hR1fgGUOjC4gwBqd1AEJECBAgAABAgQIECDQtIAAq+n2TKI4AdYk2tjWixBgtdUP1RAgQIAAAQIECBAgQKC2gACrtrDjC7DMgdEFBFijkzogAQIECBAgQIAAAQIEmhYQYDXdnkkUJ8CaRBvbehECrLb6oRoCBAgQIECAAAECBAjUFhBg1RZ2fAGWOTC6gABrdFIHJECAAAECBAgQIECAQNMCAqym2zOJ4gRYk2hjWy9CgNVWP1RDgAABAgQIECBAgACB2gICrNrCji/AMgdGFxBgjU7qgAQIECBAgAABAgQIEGhaQIDVdHsmUZwAaxJtbOtFCLDa6odqCBAgQIAAAQIECBAgUFtAgFVb2PEFWObA6AICrNFJHZAAAQIECBAgQIAAAQJNCwiwmm7PJIoTYE2ijW29CAFWW/1QDQECBAgQIECAAAECBGoLCLBqCzu+AMscGF1AgDU6qQMSIECAAAECBAgQIECgaQEBVtPtmURxAqxJtLGtFyHAaqsfqiFAgAABAgQIECBAgEBtAQFWbWHHF2CZA6MLCLBGJ3VAAgQIECBAgAABAgQINC0gwGq6PZMoToA1iTamdM4556TXve516Qtf+EK68MIL0xZbbJGe+9znple/+tXp+te//vKrPP3009Mee+yRTj311HSrW90q7bnnnmnvvfe+msKJJ56Y9t9//+6Y22yzTTrssMPSDjvsUCwlwCqmMpAAAQIECBAgQIAAAQKTEBBgTaKNTb8IAVbT7Skv7tOf/nQ64YQT0nOe85x0pzvdKZ155pnpRS96UXre856X3vjGN3YHysHStttumx7zmMek/fbbL51xxhlp1113TUceeWTabbfdujGnnHJKesQjHpEOOeSQ9MQnPjEdf/zxXYB12mmnpRUrVhQVJMAqYjKIAAECBAgQIECAAAECkxEQYE2mlc2+EAFWs61Z/8KOOOKI9M53vjP98Ic/7A6W/+/8iaz8Ca2lT2Xtu+++6aMf/Wj63ve+14151rOelS677LL0iU98YrmABz3oQWm77bZLRx99dFFRAqwiJoMIECBAgAABAgQIECAwGQEB1mRa2ewLEWA125r1L+w1r3lNyp/M+sY3vtEdbKedduo+hZUDq6Wfk08+OT360Y9OF198cbrZzW6Wttpqq/SKV7wi7bXXXstjDjjggO4x3/nOd4qKEmAVMRlEgAABAgQIECBAgACByQgIsCbTymZfiACr2dasX2E/+MEP0n3ve9/u64P5q4T557GPfWzaeuut0zHHHLN88LPOOivd/e53T/mfd73rXbtPZh133HHdVxGXft7xjnekgw46KP30pz9da1FXXHFFyv9b+skB1pZbbpkuvfTStMkmm6zfC/FoAgQIECBAgAABAgQIEGheQIDVfIvCFyjAaryF+St++R5Uv+vn7LPPTne5y12Wh/z4xz9Oj3zkI9P222+f3vOe9yz/+1oB1oEHHtgFXGv+CLAan1zKI0CAAAECBAgQIECAwEgCAqyRIB3mWgUEWI1Pjosuuij9/Oc//51V3uEOd1i+p9UFF1zQBVf5vlXHHnts2nDDDZcfW+srhD6B1fgkUh4BAgQIxXj1bgAAIABJREFUECBAgAABAgQqCwiwKgM7fBJgTWgS5E9ePepRj+q+Ovj+978/Xec617naq1u6iXv+KuD1rne97r+96lWvSh/+8IevdhP3yy+/PJ100knLj33IQx6S7nnPe7qJ+4TmipdCgAABAgQIECBAgACBMQUEWGNqOtbaBARYE5kXObzKn7y6/e1v393DavXw6ja3uU33KvNX+u585zt398LaZ5990plnnpl23XXX9OY3vznttttu3ZhTTjml+/rhoYcemnbcccf0gQ98IB188MHptNNOSytWrCjSchP3IiaDCBAgQIAAAQIECBAgMBkBAdZkWtnsCxFgNduafoXlrwvusssua33QVVddtfzvTz/99LTHHnukU089Nd3ylrdMe+65Zxdmrf5z4oknpvwXDM8555y0zTbbpMMPPzztsMMOxQUJsIqpDCRAgAABAgQIECBAgMAkBARYk2hj0y9CgNV0e2IWJ8CK2TdVEyBAgAABAgQIECBAYKiAAGuonMeVCgiwSqWMKxYQYBVTGUiAAAECBAgQIECAAAECIwjYh46A2PghBFiNNyhieRaOiF1TMwECBAgQIECAAAECBOIK2IfG7V1p5QKsUinjigUsHMVUBhIgQIAAAQIECBAgQIDACAL2oSMgNn4IAVbjDYpYnoUjYtfUTIAAAQIECBAgQIAAgbgC9qFxe1dauQCrVMq4YgELRzGVgQQIECBAgAABAgQIECAwgoB96AiIjR9CgNV4gyKWZ+GI2DU1EyBAgAABAgQIECBAIK6AfWjc3pVWLsAqlTKuWMDCUUxlIAECBAgQIECAAAECBAiMIGAfOgJi44cQYDXeoIjlWTgidk3NBAgQIECAAAECBAgQiCtgHxq3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179+d/h99903ffSjH03f+973uv//Wc96VrrsssvSJz7xieWnf9CDHpS22267dPTRRxeVZOEoYjKIAAECBAgQIECAAAECBEYSsA8dCbLhwwiwGm7O+pR28cUXp9133737JNZXv/rV7lA77bRT9ymsHFgt/Zx88snp0Y9+dMrj89cOt9pqq/SKV7wi7bXXXstjDjjggO4x3/nOd9ZaUg7I8v+WfvJzbLnllunSSy9Nm2yyyfq8DI8lQIAAAQIECBAgQIAAAQLrFBBgrZMo/AABVvgWXv0F7LPPPumoo45Kl19+ecqfnMqfpLrFLW7RDXrsYx+btt5663TMMccsP+iss85Kd7/73VP+513vetfuk1nHHXdces5znrM85h3veEc66KCD0k9/+tO1ah144IHdf1/zR4A1scnl5RAgQIAAAQIECBAgQKBRAQFWo40ZsSwB1oiYNQ6Vv+KX70H1u37OPvvsdJe73KUb8l//9V/dp6nOPffcLlTadNNNuxBrgw02qBZg+QRWjc47JgECBAgQIECAAAECBAiUCgiwSqXijhNgNd67iy66KP385z//nVXe4Q53WL6n1eoDzz///O6rfPkm7vnm7rW+QrhmcRaOxieV8ggQIECAAAECBAgQIDAxAfvQiTV0LS9HgDXhHp933nnp9re/fcr3udp+++2Xb+KevwqYb+qef171qlelD3/4w1e7iXv++uFJJ520LPOQhzwk3fOe93QT9wnPFS+NAAECBAgQIECAAAECkQUEWJG7V1a7AKvMqflRX/va19Kpp56aHvawh3U3Y/+P//iPtP/++3f3rfrud7+bNtpoo+6m6ne+8527rxLme2WdeeaZadddd01vfvOb02677da9xvxprUc+8pHp0EMPTTvuuGP6wAc+kA4++OB02mmnpRUrVhQ5WDiKmAwiQIAAAQIECBAgQIAAgZEE7ENHgmz4MAKshpvTp7QzzjgjvexlL+v+UuBll12WNt988/T4xz8+veY1r0m3ve1tlw91+umnpz322KMLu255y1umPffcswuzVv858cQTu8edc845aZtttkmHH3542mGHHYrLsXAUUxlIgAABAgQIECBAgAABAiMI2IeOgNj4IQRYjTcoYnkWjohdUzMBAgQIECBAgAABAgTiCtiHxu1daeUCrFIp44oFLBzFVAYSIECAAAECBAgQIECAwAgC9qEjIDZ+CAFW4w2KWJ6FI2LX1EyAAAECBAgQIECAAIG4AvahcXtXWrkAq1TKuGIBC0cxlYEECBAgQIAAAQIECBAgMIKAfegIiI0fQoDVeIMilmfhiNg1NRMgQIAAAQIECBAgQCCugH1o3N6VVi7AKpUyrljAwlFMZSABAgQIECBAgAABAgQIjCBgHzoCYuOHEGA13qCI5Vk4InZNzQQIECBAgAABAgQIEIgrYB8at3ellQuwSqWMKxawcBRTGUiAAAECBAgQIECAAAECIwjYh46A2PghBFiNNyhieRaOiF1TMwECBAgQIECAAAECBOIK2IfG7V1p5QKsUinjigUsHMVUBhIgQIAAAQIECBAgQIDACAL2oSMgNn4IAVbjDYpYnoUjYtfUTIAAAQIECBAgQIAAgbgC9qFxe1dauQCrVMq4YgELRzGVgQQIECBAgAABAgQIECAwgoB96AiIjR9CgNV4gyKWZ+GI2DU1EyBAgAABAgQIECBAIK6AfWjc3pVWLsAqlTKuWMDCUUxlIAECBAgQIECAAAECBAiMIGAfOgJi44cQYDXeoIjlWTgidk3NBAgQIECAAAECBAgQiCtgHxq3d6WVC7BKpYwrFrBwFFMZSIAAAQIECBAgQIAAAQIjCNiHjoDY+CEEWI03KGJ5Fo6IXVMzAQIECBAgQIAAAQIE4grYh8btXWnlAqxSKeOKBSwcxVQGEiBAgAABAgQIECBAgMAIAvahIyA2fggBVuMNiliehSNi19RMgAABAgQIECBAgACBuAL2oXF7V1q5AKtUyrhiAQtHMZWBBAgQIECAAAECBAgQIDCCgH3oCIiNH0KA1XiDIpZn4YjYNTUTIECAAAECBAgQIEAgroB9aNzelVYuwCqVMq5YwMJRTGUgAQIECBAgQIAAAQIECIwgYB86AmLjhxBgNd6giOVZOCJ2Tc0ECBAgQIAAAQIECBCIK2AfGrd3pZULsEqljCsWsHAUUxlIgAABAgQIECBAgAABAiMI2IeOgNj4IQRYjTcoYnkWjohdUzMBAgQIECBAgAABAgTiCtiHxu1daeUCrFIp44oFLBzFVAYSIECAAAECBAgQIECAwAgC9qEjIDZ+CAFW4w2KWJ6FI2LX1EyAAAECBAgQIECAAIG4AvahcXtXWrkAq1TKuGIBC0cxlYEECBAgQIAAAQIECBAgMIKAfegIiI0fQoDVeIMilmfhiNg1NRMgQIAAAQIECBAgQCCugH1o3N6VVi7AKpUyrljAwlFMZSABAgQIECBAgAABAgQIjCBgHzoCYuOHEGA13qCI5Vk4InZNzQQIECBAgAABAgQIEIgrYB8at3ellQuwSqWMKxawcBRTGUiAAAECBAgQIECAAAECIwjYh46A2PghBFiNNyhieRaOiF1TMwECBAgQIECAAAECBOIK2IfG7V1p5QKsUinjigUsHMVUBhIgQIAAAQIECBAgQIDACAL2oSMgNn4IAVbjDYpYnoUjYtfUTIAAAQIECBAgQIAAgbgC9qFxe1dauQCrVMq4YgELRzGVgQQIECBAgAABAgQIECAwgoB96AiIjR9CgNV4gyKWZ+GI2DU1EyBAgAABAgQIECBAIK6AfWjc3pVWLsAqlTKuWMDCUUxlIAECBAgQIECAAAECBAiMIGAfOgJi44cQYDXeoIjlWTgidk3NBAgQIECAAAECBAgQiCtgHxq3d6WVC7BKpYwrFrBwFFMZSIAAAQIECBAgQIAAAQIjCNiHjoDY+CEEWI03KGJ5Fo6IXVMzAQIECBAgQIAAAQIE4grYh8btXWnlAqxSKeOKBSwcxVQGEiBAgAABAgQIECBAgMAIAvahIyA2fggBVuMNiliehSNi19RMgAABAgQIECBAgACBuAL2oXF7V1q5AKtUyrhiAQtHMZWBBAgQIECAAAECBAgQIDCCgH3oCIiNH0KA1XiDIpZ36aWXppve9KbpRz/6Udpkk00ivgQ1EyBAgAABAgQIECBAgEAggRxgbbnllumSSy5Jm266aaDKlVoqIMAqlTKuWOD888/vFg4/BAgQIECAAAECBAgQIEBglgL5gxS3u93tZvmUnmtGAgKsGUEv0tNceeWV6YILLkg3uclN0gYbbDDaS19K1KN8sitavblR0WpW72hvr2s9EOO6xnzr+lrX+K5NwPuu7rzgW9fXusbXunbtc+Cqq65Kv/zlL9MWW2yRNtxww/qTxTPMXECANXNyTzhUINp3mqPVu3RBlD9um78GGuHrn9GMo9VrTgxdrcofF21ORKvXHC6fi0NHmhND5cofF81YveW9HTqS8VC5ssdF83WuK+urUdMQEGBNo48L8SqinUyi1evkV/9tZE4wXlMg2pyIVq91zXtubQLR5rF6687jaL7WtbrzIaJvxJojvu/qzzzPUCIgwCpRMqYJgWgLXbR6nfzqT3NzgrEAq/4cYDxbY+tafe9oxuo1J6zD9ecA49kbe8Y2BARYbfRBFQUCV1xxRTrkkEPSfvvtlzbaaKOCR8x3SLR6s1a0mtVbf44zrmvMt66vdY3v2gS87+rOC751fa1rfK1r9eeAZ2hXQIDVbm9URoAAAQIECBAgQIAAAQIECBAgkFISYJkGBAgQIECAAAECBAgQIECAAAECTQsIsJpuj+IIECBAgAABAgQIECBAgAABAgQEWOYAAQIECBAgQIAAAQIECBAgQIBA0wICrKbbozgCBAgQIECAwPwErrrqqrTBBhvMrwDPTGA9Bczh9QT0cAIECDQkIMBqqBmLXMr//u//phve8IZhCKLVm2Gj1fzLX/4ybbzxxssbp9YvQKPVa07UX26iveei1WsO15/D//3f/92tw9e73vW6J2t9Hc41RluL1Vt3HpvDdX0jrsPOdfXnRLR1rb6IZxhTQIA1pqZj9Rb41a9+lV760pemc889N93qVrdKe+yxR3rgAx/Y+zizekC0erNLtJpzvXkenHXWWemWt7xl+uM//uP0rGc9a1Yt7v080eo1J3q3uPcDIr7nIq3D5nDvKdn7AXkO77777unUU0/t1uHHPe5xae+99276k1jR1mL19p6WvR5gDvfiGjTYuW4QW68HRTSOdA3fqxkGNyMgwGqmFYtXyE9+8pO0ww47pBvf+MZdSHHMMcd0F8f5/84XyldeeWXacMMNm4GJVm+Gi1Zz/k3pE5/4xG4evOQlL0nHHnts+o//+I/0pCc9Kf3VX/1VM3NhqZBo9ZoT9adQtPdctHrN4fpz+PLLL09Pf/rT0y9+8Yv0qle9Kn3kIx9J//RP/5Qe/vCHd2vyda5znfpF9HyGaGuxens2uOdwc7gn2IDh0c4d0ep1rhswKT1kYQQEWAvT6vZe6D/8wz+kAw44IH3mM59Jt73tbdOll16ajjzyyHTooYemb37zm+lud7tbU19XiFZv7ni0mr/4xS+m3Xbbrdsw3f3ud09XXHFFOv7449MLXvCC9KlPfar7FEBLP9HqNSfqz55o77lo9ZrD9efwGWeckZ761Kem97znPWn77bfvnvDkk09Of/AHf5COOuqo9MIXvjBd97rXrV9Ij2eIthart0dzBww1hweg9XxItHNHtHqd63pOSMMXSkCAtVDtbuPFLn2y6u1vf3s65JBD0vnnn79cWP4NyfOe97wuuPjKV77SRMHR6s1oEWvOdX/oQx9KO++8c7rsssuWe5/vuZLnxOmnn56+/vWvpxvc4AZNzIto9ZoTdadNNN9o9VrX6s7f1Y+ew6rHPvax3Tp8/etfP/32t7/tPnX1yle+Mp1wwgnpC1/4Qtpmm21mV1DBMzl3FCCtx5BovubwejR7HQ+Ndu6IVq9zXb2568jTERBgTaeXTb+Sd73rXd2NYO9///svX/jmf5e/NviWt7wlPexhD1uuP39VYccdd0yf/OQn02Me85i5fAorWr0ZL1rN+dN2+cbA97nPfdKDH/zgrv+f+9znuq8O5mDzaU972nLvf/CDH6R73OMe6X3ve196znOeM5evl0ar15yovyRGe89Fq9ccrj+H81qbN3grVqxIT37yk7sn/Pd///fu3Lvvvvt298H6zW9+033iKv9zs802677iv88++8xlHc71RVuL1Vt3HpvDdX0jrsPOdfXnRLR1rb6IZ5ilgABrltoL+Fz564H50zP5K4L5ng/5t7n5gvjlL39594maZz/72en5z39+2muvvbr/ln9++tOfphe96EXpZje7WTruuONmqhat3owTreYvfelLXd+32GKLbkN04YUXpl122aX76uh5553X/d93vvOd0xFHHNHdHy1vrvL/8pz4z//8z5S/ejHLn2j1mhP1Z0e091y0es3h+nP4a1/7WnrGM57R/fGUTTfdtPvafv7a4Jve9KbuFwsvfvGL08UXX5z+/u//Pt30pjft/hhIPkfne2LlT+P827/9W/0i13iGaGuxeutOEXO4rm/Eddi5rv6ciLau1RfxDPMQEGDNQ32BnvOP/uiPur9gdPTRR3cXvB//+Me7397mf+abdecwK184H3744cv32sg8+XE3uclN0l//9V/PVCtavUtWkYx33XXX7iuif/d3f9eFlV/+8pe7vzL41re+tfv01WGHHdZtkF72spel5z73ucv9//M///N05plndvf1yp/mm9VPtHrNifozI9o6Ea1ec7j+HM6/RMp/ICOfi//f//t/6Vvf+lZ3Ts73t3r961/f3Ycwn5fz3MmfxMpf5c5/XCOv0+9973vTpz/96bT55pvXL3S1Z4i2Fqu37vQwh+v6RlyHnevqz4lo61p9Ec8wDwEB1jzUF+Q5f/jDH6b73ve+6YMf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nIj2notWrzlcfw4vrZl5Dc43X883Wl/6yfeczPca/MlPfpJOOeWUdJ3rXGf5v61+D6c8r/L/n9ftpRu3W4tXCUQ7d0Sr1xyuv05EO3dEq9e5rv4crnlOcux4AgKseD2bScX5wvcDH/hA9xWCfN+Mm9/85t3z5gAr35A73x9j6d5VOYTIXyU87LDDurDiHve4RxfG5N+mnXjiielv/uZvuseu/tdCxn4R0erNrz9azbne/LWUPBfynLjjHe/YtfEtb3lL98mp/NXBpfte5X+fP6mXv1L4la98JT3oQQ/q+p8/uXf44Yd3XyudxZyIVK85MZs5YV27svv6WK0f61r9c11eczfeeON073vfOz3gAQ/oWpnX2xxg5b/m+vjHP375fHvGGWd04z760Y9296fMfwAhh1l777139wc1bnGLW1ztFww15oVzR/05Eelcl+eDOVx/TjjXOdetvp5HW4drnIscczoCAqzp9HK0V5LvS/Wnf/qn6U53ulN38/UcSOWvIeTf5OaAKn8VMH+U/klPetLyc+abt+evITz5yU/uLozzT76Qzve6yo956EMfOlp9ax4oWr25/mg1583PLrvs0s2JCy+8MN3oRjfq7mGV58nXvva1tPPOO6fdd9+9u+fV0m/586eqlh6Tw638k7/ekjdR+SbCj33sY6vNiWj1mhP150S091y0es3h+nP4M5/5THdPq/zLg/x1/Isuuqi7d1X+a675E7AvfOELu7Aq/5Ig34cwfxIn33sy38w9j8+/VMo/Z511Vrr//e/ffWI6f1K25k+0tVi9dc/N5nD9dSLauSNavc519edwzXOSY09DQIA1jT6O9iouv/zy7kawj3rUo7obr+e/Evi3f/u33SdszjzzzO6vCD784Q9P17ve9bqvja3+FwRz0JUvrvfdd9+unvxVwfy1hJp/ZTBavdklYs35xv353lT5Jvz5ppof+tCHuhsD54vRRz/60d29rPINg/On8B7ykIcsz8f8SYBtttmmCzPzzdvzJir/Nazb3e52o83ZtR0oWr3mRN05Ec03Wr3Wtdmsa8961rO6r2XnTzznrwXm9XfXXXftPhmb/5n/IEb+d/mTWPkXTks/+RcN+QbZ+Tyef/mQg60cfuV7Vtb+ibYWq7fuudkcdq5bfc1xrqu/T8re0da12uclx48vIMCK38NRX0EOrHIgkUOKfCPY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qEtD4aYqrlsvZxOkurCBIjqA6K2QLTa9sol8bELjWshep3Et9Ljmx1YiqIhsffHFFxse6IIFC1w09BZbbGFTpkxpqEHJC/wNsPw39a+IwuJwikja0C22sVh6w87NsmHNdaVjTmzrYa3Xwttw6HlJ71+/BOTAakXPHifTH//4RxchRTqYT+nae++9Xd0rboTzETU4H0g7ILyeKCwcXMuWLXML5tdee83dJBe6xaYXHrFppqgq9VJ4vtttt52LpqMNHjzYbXyuu+469zvakiVLnOOKNBZuFuzSpYuzCQoHT58+3caPH+9el3RsVdtGYtMrmwhvE7H1udj0yobD2zDjGlFV1LhiHO7cubMbOqlnRQ3KX/3qVw1OKMZX5mFSt5mfSeNmHL733ntdqjfRWqHHYd4/trFYesPPzbLhsIxjmzti06u5LvxcV+09gd5PBMoRkAOrldgGC2DqEXFrHA6o/v3724knnmg9evRwTgqKP5I6yKmubzg2fvazn7kaV9Q9YuGMY4Mb5aZNm+bqcoRqsemFQ2yaWWxSWJXUzzlz5rhoqTPPPNPVWsFpSU0VbhbcZ599Gh4zxdipg0V6iq+5QkrLmDFj3I2DOEZDtdj0yibC20RsfS42vbLh8DZMijXzKo4oUv64sff88893qdik6RP1yu2uXJziG5dmHHXUUa5G5dixY92P77zzTjv00EPt8ccfDx4NHdtYLL1h52bZcPhxIra5Iza9muvC23CovYHeVwTSCMiB1Qrs4q233nJF2XFYHXvsscaNRDfffLPNnz/fObNonTp1co4KImy4idA36m7wWtIMadRC8jcVhkITm144xKaZuij77ruvHXLIIc6xiVOS58xpP84saqtw0xXRepzqJ+uaUdeKAv9srGikkBIB4G+8CmEXsemVTYS3idj6XGx6ZcPhbZgbXomA7tmzp1100UXuwhQODbjdlyjXXXbZxc29pGVfeumly6Vk8zff//733eEB7dNPP3X/uDwjZIttLJbesHOzbDj8OBHb3BGbXs114W045Jyk9xYBObBaqQ2wIOb0lgKwvhA70Vac1hJZRUrY1KlTXVQWta+OOOIId6scp7y+3hE30uXVYtMLl9g0k0pKHRWcmN/6/9fAU4gfJyapgXfffbcr1r7ffvvZWWedZcOHD3c3TVJThag9foYzNK8Wm17ZRHjLiK3PxaZXNhzehp999lnbc8893S213bp1a/jA3Xff3dWn5GCBW9yYm4m28mn+OA0Yh5nDOYDIs8U2FktvWOuQDYflG+M4rLkuvE3ENq6FJ6JPEIHlCSgCqxVYBIW4ufWDYrCkENKImPn1r3/tomi4zYgILBwS1NQgpYzbBYnGwulFfSMicvJqsemFS2yaX3rpJevVq5erl0LqoC9oy//HaUXBdk73SWXhuxGJN2TIEHejIBsqbCn0zYJJe4tNr2wi/GgRW5+LTa9sOLwNc2jAgQF1BImI9UXYiXT+7ne/66Jijz76aHfByk033eTqTnEYRWohYyIRs6Qe5tliG4ulN6x1yIbD8o1xHNZcF94mYhvXwhPRJ4iAHFitzgZIRfApX6QR+vb2229bv379bKuttnIFuD///HNXR4OFMv9N2hhOrvXWWy9XJrHpBU5smklVof4Vp/o+BYXvwck+NbDatm3rHJcUbn/66aftsssuc2ks66yzjksppNhwni02vbKJ8NZR1D5X7hKDoupt7EnFpjm2cYLN/8knn2zt2rVzhwN+DOaCFWpckc5NhAvjMhHR1KR8//33bdVVV3UHTOuvv374jlbyCbExLrLetLGiyHrTjE02HL4LFnUc1lxXu31SbONE+F6iTxABObCiswEWt/5GwXLiOd3lhkHqaPhC7Uw+pIK98cYbNmnSJFtttdXcn3P9Ns6KPKOuSnUXTS9Rajj02Gj4aKUia2bT88EHH9gPfvADa9++fapZEFH1+uuvu9orRFv570VR/9/85jfOgeUdVfyO+ioUGA7RFi9evJxTLI1xkfQmN5qN8SiaHTf17IrEmBuMytmu/x5F4sslBl27dm24yTONdZH0oo9U8nHjxtmoUaNcFG7RNWeZ64pkw031N37PJRhcmMJFKn369HGpg6RrM/4SCc3FGclnk6VfZPnccq+hZlRynE/bpBaJMZt7bspt7FKZIunlQIj+dsUVV7jDw7RWJL2l9pZmD0Wz4Sz2X2TGRR+HNdeF3ye1xrkuS7/Ua0SgWgSUQlgtkgHeh/Q+ahMRLUMB7dNOO81FyCQbqYL8nkUei2EisHBasUCmccrLzUekjIVu6OVacDalFJodNmyY01ZUvV7XyJEj7ec//7lbcHIbVGkrEuP33nvPPWdqp4wYMcI934033ng5yd459Nxzz7nU0m222cad5uOco2FTDzzwgP3lL39ZrqB/CPvAJtBAdMHaa69t+++/v7vlMNmKpBddsfU7nIPUuNtrr71syy23dLeJUuOuqIzhe8455xgbJ2yC2y5LN3pF6nNcYsC4isOB2nFcmFHkMQJtaOYm0QcffNDZwssvv1xoxtgEtZ6YOzhYIXrU1+7zrIs0TtDnSPsjTZvDgcb6HCnZjNUcGmE//iDpqquucjfbPvHEE64fhG4w5hbaDz/80H3ekUce6cbjoo4TyX5HzZ3DDz/8a4iKZBPoJR2U9E8a4wUO7aLyxR6Yi9lIE4U/aNCg5S4RQLfnWxQb1lwXdpTQXJfPPimmuS6sxendRaD5BOTAaj67oH+Jc4GNEilg1Ky64YYbbKeddnLOB4rC+g1eUoR3TPTu3dulFFKQ+7jjjnOF3Fm4hmykoI0ePdoVhWeBPnnyZLcx5SSy3OK8lno9CzbR3BLVpk0bVweKTQV1xMpFYdVS82effeYK8KP1yiuvtM6dOzc80nKh3hTwZ6OF7VDvis0sNVaIJPHXs4eyC4pQUneNKDEW9lwg8Oqrr7pLBXbbbbfUj62lXgTF1u9IB8YmiarAScmz9c7rcs+1loxxFGN39DlftJqb2Oh3RFmktVr2OT6bG+MYU4lczHIDXC31wo+DDrRSABzHCbfZkTqOg7OI49rvf/97Gzx4sHMEUdwcO6b+HnMY6c5pJ9W1tOEJEyY4B+y7777rDjyY93C8lRuDeSb33nuvXXDBBS7KlwMTUgM54MFZRzRsqcO52mMykdlE0WAD2DKOMyK2sW0OONJaLRnThyhoz/PP2u9qqZdDGtZo9DlS9hnbLr/8chs4cGDZPldLvX/605+cg5uLAlhnop31BRtrHJtp40StbVhzXdg1vOa68Puk2Oa6as9Dej8RqCYBObCqSbOK78VCgtQ/TmxpnKCTCkYkC+ljODH8gtkvNnBY4TgiAgunF+ljLKT4WagTXjQQ/UGaCrU+uE2Jds8997gTvRdeeOFrxcBrqTf5iLwOFnDbbruti2piM81iv7QVQTNRdpyY4hhiA8TJPUWBibrx0QqlNoGDjsUqz4K/oS4aRd2xidC1z3BaUcuFmlq02bNnuxSa22+/3bbbbrvlEHu+tdSLoFj6HVqpn8MmH2cQ0XZEWuLYxFmZ1mrJGLv8wx/+4JyXpHZgBzTGDsYNHJul9ljLPodeoj6IcGRzh4OFtmDBAudoW2mllQo3RtB3DjroIJdafM0117jNKRELRIH07dvXOVCKOK4dcsghLu0d5yaNeeuEE04wok0ZM5jrvC3U0obRRj0gnMT0sY8++sg5vDkcgm9a8+MxB05coII90fhuHCJgY6HrXKERJxuHCThYaMx1HIYQLcT8kTYv1mosJlocRybzMJtqGkXvGR/WWmutsjZcC73Mv4xfFFzG4Uaf41kzJpMWinOyXJ+rhV6vhYNNDhpxFnt75MARe6DPJW2yCDasuS7sGl5zXdh9ku93Mc11qROafigCBSIgB1aBHgZS/GKBgY6F+5QpUxoUkvblnUQsSMudphPuvXDhQncqzIIqdONEGmfbeeed11DTBmcaCzhO/onKaqzlrTephU0/m2ly/i+++GJ7/PHH3SZ7++23L8uXv89Ts7cJNqW/+93v3KaJBTLh3iygO3To4BxUpN2UswnSBRYtWuR+X87BUS07QS/1tOC66aabNiziudmQU2kKFXMzFxuSIuiNtd9R7Bl7JaqCSxl23nlnd3qOA5saN41FhPCd87QJvwklfYm+5R1AXC5BzRjGDp8KW053nn0OvThSiLBBK+nFjL04MLgpjssxSCskCrIoej1jXwPR92cibA444ACX3l2uv/nX5s2Yz+MAAScQTisfbcX4RvojtkzkUFHGCZwT6OIQgPn54IMPdmMcUTdE55WzBf9zIiVxKtJfOTTJo2Gzb731lrvN0NecoyYXBwkXXnihizjGxovCmEggnNzz5893F84QUQgz2GEXOJMZ62qt138+a63SNH6i3HC2TZw40dlJYxF2eY/DOLgZD2DJmsf3ORyat912mytRwOU+pXxracOxzXWMa3DWXBduhMOpHdNcx3qCA6ZY5rpwT07vLALVISAHVnU4NvtdqIvBJpQIGqKmVl55ZbdwIHqFhQUnZD4ygUUvDgAWRUTf8DdNbVKbLazMHyb1cnLLgjhtIcmmlFNpIsdCRX9l/W5JzdTaSUZPsEhmQcy1wJyi4lzBIYRjiLQA0t/ybuX0stnnpGyzzTZzDiLSNnFioZ2FKLeWcDV7U5vUan+fNBtmo0fKIifQpCqw2CACh3QmoljWXHNN59zku+Stl+/PZogNHCfN9Dt0oBmtRex3OIQJP2ej1L1799RIBJzaRNaRjrfffvtV+zFX9H5N6WWTOmDAAJs2bZpzBtAPGfNIeWvMsVmRiApenKaXsZVIUpyCjBk4rHD+Ysv0xV133dVFXXTs2DH3cZiv1hRjn2aOM4DoH8bkWrY0vVwowiYEBzwOQmo84mxhTMYxRAQy2msx1zHf4lzwY0NpPUdY4mAhQg/bgHOtW1Oa+T1RYNg1tky/I52bVELqbOY9FpfT++KLLzpdOH+IeMNJzyEZBzj8jMhNxu681z9JvWmprb7PES1/8803u7muli2NL7pxXjHv4aDHvokIJEKP9RFRvFxqQ//Lmy+s0tYTpQyLNNcl9bL+8rVGveaizXVpeos+1zXFuGhzXZoNs6fgIooiznW1HKP02SLQXAJyYDWXXBX+jlNbnBBMeqQAMbDhNGFhwQkYDgA22UknCiHenEpTP4Mw8Dxbml5S2tiAMAHyj8UljSvDiWQisomf+Z/nqZfPakwzvyeVgs0HpyM0FnaPPfaYq01Cqia39IWuT5JkkqaXyCpshM0+p9FsBKlhQhQIjdNJUjfZ/GEvebbGbJiTfxg+9dRTrgYMNr3jjjs6BwDfA66kQ+bd0Ew0Co4TtLBQpy4TNXhwuBENVKR+Rw0Y+ny3bt1cegeOEzTiDL7y76wAACAASURBVGTDSb9jc83/kppJZAeOWFLdarEBaUovY8Frr73m0jVJa2MDdd9997k05H322celO+XZ0vRS44g0JqIWqd2GU40NHjbLP/ofjgted9hhh+Up131WFsZeFM5txo677rqrbJ2x0F8gTa+v50hkKX2SzTN9kRtzicYiMotxgr7qU05D6/TvjyZqVTH34mjnFuBkX/KOHn7GOMwGCrsl2q0WfQ7dTWnmNYzJMGU84QCKsZk6lUTEwj3PlqbXc8WxyWENm38cm/4WZg5BiBrD+R26rmcpiyx8/d8wZqCTQwci32rR0vT6GzCZ30jZhTcHIkT6s4bALojEwoHMuiPv1tQaE71Fmuua0stcUaS5rrH1JYeh2C1r3yLNdY0xLnW4F2GuS9NLei7jAH2S+Yw9XlHmurz7uD5PBKpFQA6sapGs4H2I/uBKbaKoGOxI82NzwQafW4FwXNHw1rMh5XTU1yRgAULaCn/n62lU8NHNemljeimyymbaN38SQiQLkwsTom/+uuY8FvhZNbNxprYUTgGccSyUYc2JP6mRRArlcSqd1SZILcVWcBCy4PTaOP0nZJ1NiE8TadbDzvhHWfXydmw2iMLDoekbJ9RsDEnHYcGUR+NUjM3QK6+84hYR9DXqrWC/OE7YsBap36GFk3GiqXBuDx061NUzw8mGk5W6UaQK0nw0ABsTNiGMD0QOsQEk5S2vqKYsesv1fxxGODGIavIb1tB2UU4vdkHdFS52YGFPOlAy2pQNNs5t6rth33m2LIzR48cGHAHYDvXviC7Nu5XTy+EGqTbU4iGljf8Pa7RyUQmN8RenBc7lPBr9iLGJqFbmAcY5NvNEB5barZ/rZs6c6SIHcbIxhhAdy99RsD2PuSOr5nJaKDTuI1KpWRi6NaY3qZGDGqJZ+Jes7ci6iKgsxrk8Wla+SS0czOAIevLJJ7+W5hRac2N6k5f/sP5kPiEKj7RBxl8aFyiw5szTaZx1vYY+/x1qOddl1VuUuS7reo0DG9Zq9LmkM7kWc11WxtiE51zLua4xvazh/PqSuY71BaxrOdeFHof0/iIQmoAcWKEJp7w/JzLU9aB+Bv98o9gunnlSgGjUvOJ0H2cXp42c5lALgtB6Ts98QdbQXyGL3uREzaaZk2giFPhOLOaINGMjkFe0QhbNcOPkkeu52SjhwPJRb+jl9kY2M3m0rHpJp8GJRdQKUU2knNKouUFKCKfpebQselkALVmyxC3kqVlCtIpv2AGpV9hIXg1HDjU+qPmSvAWRzTHRE34Bj73ilKh1v4MLTj5OQ3E+eGcVKVbUleM7kMpW2qhvQuN7kW6MHYe+cdJryKq3dDNNqhY2TDQkTsa8WmN64YcTO61Rh44UaRz0pDfl2bIy9proq3wX0q9w2uZxgJDk0ZheDg6wUVqpTZDmRhQIziQcm3lFwvLM2YzAighB0rBwGrOJK+cEIiIazozBbKLgTTpvXi2r5tJnj5OIuiz0vbPPPjsvua5fVcoYcTDm0Iz5mXkwr1apXgr8M8+wluOgMQ9HZpJFY3rTUh/937Kx5uCDA7I86qcmx6im1sRp41at5ros6x+vt1R3Lea6LHrL9aVazXVZNJeyreVcl0WvZ1yqu1ZzXV7jpz5HBEIQkAMrBNUm3pNIJOotceMdLRm1xIk/YaZ+gCPNglMyolSIwmDDyvXdLOr9zXOhv0IWvUkN3DzIyQKRFDjapk+f7m5LY4GSV8uqmagl6mngEOBkl4YDjtNdogDyWiRn1Ys+nD7YCI1IN6JxiGRgw0RaXB6tEr2clmMTbFbZLGEXpMJRP8ZHWuSh2T9bH4WCc41UJRxW3NaJQwit9EHslci8WvY79LJ4ZMPGSX6yvhm1gXBg49DEWZxMJaT2Cifq1G1i/CCsPq+WVS96fNox6bv0Nza0OIRKCyKH1F6pXrQki7vT51ZZZZWQEr/23lk1+3mFSDLvjMeZnHfLqtfbBHZA1BYHHqS2EZmcZ0tGqeCMInIUfkkHvNfjHRM4xmHLgQJ9Lq+DD6+jEs30OyIIudEYpxUbapyIjC95tebopVg64zK1ZODNAUherRK9aCJVk8NJnKB5HSolWVSiF3vgYJQIZSK5icDB7ldcccW88Fol6wlE+e9Xq7muUr21nuuao7fWc12lmtFby7muUr3YRK3nutw6uD5IBAIQkAMrANRK3tIvgPlfFussfElX8LUKGBQp8k5hTRYYFFolbSXtOulKPre5ry2nN3nCiD5yvmmc7LLIqGUh93Ka0070/GuTC8Dmsmru3zVlE2ijoC0OQepfEZlDyH+eC/rkd2tKL05CUvaIKIQ50RRsCmulF+2ckJM2SFoYzlZq0OHIwnlCvSCcmGy8a93v0EWh3WSKI/rZgPbq1cvd9IkTBa48BzZ4ODdxbHL6T2Rhni2rXvRTR4pLHoiCxEmHY7aoenF2UscJvUQU4YhFL87OvFtWxsk6TaTDYcuki+XdsuplXONgB67YNNEVOK9IMaxV47kTmcvGiDGWekGl0TT0OX5HpA3O+bxtuJRNY5qJviHNHIc4qSxESNMPa6m5Kb0c2rEGwmnPegK9Phq1FnaRxSbQxTxHZF6eEaVpPJrSSwoh8zNRV4zDHCzVkm+WNSZjRa3nOs+6Kb2s5VkT13quq0Qvl0cVYa7Lqtm/jnVQLee6rHrRyeFjkea6Woyl+kwRaAkBObBaQq+Rvy0XMu5PYkqLmlPPhk0qIdzUuCptTNikMoTaMFVTL1fOs8lj0qbwdKhWTc15hPhXU69niiPApxFWm3M19WL3OGBxyPp6btXWy/s1ppnfk4bkX0PINxcQ+L5IbTEuRuAGLBxDtND9Lskg6VBNOlBxRhElQTQjRfC9ftIC2dTNmDGjYcNBJAXjRzJFMgRn3rO5eokexQFPZClRbkSSUXg+dGupXhybOGFJseF69DxaczUnbQKdOJHzOERorl6idHGiEHmFdgre5m0TaPdjRHIswUY5WCLS0Ue+JvsndsG8nEefK+13lWj2qWPUPuIf6w0ccqFb0iaao5eLVlgXEQUXg034OZn5LtR6rdy8UQlfb8OsK6lF6W+4DW0PaTaM7krXxLWa65qzhmdMw4ZrMdc1Ry/lKRj3ajXXNUezt+dazHXN0UuUJmugvOa6PPq1PkME8iQgB1aVaROWz0aTtBI2wZze+pasPUA9Hl7jB102z9R1IOqDxpXn3BDEjUEhWzX1cqpLtEjoVk3NsTGW3nTrqtQmko4u/9/c3EfkB4v5DTbYIGjNHfQSCYEDjXpmSYdIaY0SHGqkOlJHjMg1XwuIk/1Zs2Y13PQZskZQNfWyMA6dnhKbXqy6mpr97a+x2AROitCF+7PyLZ2beTbUwmJcII2QjTYpbHnc2ldNzaSFUQctZKtnvXnYRDX5EnGVvGQnlF00V3Mt18RZ5uYsa3jW88k9QAjG1eSbh95K5rosjPNaE1fLJvLQG8LO9J4iUDQCcmBV8YnggKIYNGH5nLwR+s7PKMCeLOhI6DMpCYTw+1t/iLKg3g43olHImMmbU19/c0UVZTa8VSi9aal51dIfSnO19JW+j/SGtWF4t4Sxf17UNWKhSYQT15+HbIwRl156qSsaTwroggULnGOqX79+DafQ9CGcwYwTnN6SDkhqBwXDGU9wclETjzTC0DVWpDf8xQhiHJZxVr6lc7Ofy+iHzMvPP/+8S+/nYgX6Y8gWSnOo+Vl6w9pEKL5FtGE01WJN3FzGTektWp+rlV4+NxTjUHYcm95QHPS+IlA0AnJgVemJcArCBvSnP/1pw7XqbIgJ1X/22Wfdp3DTBItgIi4uv/zyhltePvzwQ7dxpu7A3LlzXUFp6kZxDXeoFptef2ojxrKJZJ9oiR2/+eabrvj9zJkz3WaUq45D12EiVZiFOREdpMRwwjhmzBjnwMLhzc12hO9ThJ1brKhz1rNnT/v000/dZQgUmacOHtpJWeIEPWQ9POkNyxdbFuOwjCvlm5ybeT5cmkI/xIncv39/V0ss9AUqsWmW3rA2ERvf5oxrtV4Tt4RxLdbwsemVTYTf14XaL+p9RaCIBOTAqtJT4ZrtESNGuBNaf4sWV4BTN4PCuaQLku/81ltvuQ1s27ZtGz6Z6A+u3d5www2N+lFEVYRusemFR2yapTe0FbfMJgiXx1GME4soJiKiQjduBCQUfd68ea4QO/XLKHLNjaTcWEWBYorqoql0nEDbokWL7O2333Y3DOZRD0Z6Q1uEuVsiZRPhOLeUL6lhREZTUD6PuRkSsWmW3nD2G6M9tFRzLdbELbFh6c1m/2KcjZNeJQIi0DQBObCaZvS1V3CDGak/1MnxKYBsOomYoGYVN7mMGzfORVwRWcVrObXl59R+KQ0nZiNLzSsiLUK02PTCIDbN0hvWhkPYBLWvKI6O4zhES7MJIqaGDx9uLHg7dOjgPpa6NFx4wBhC/Tyir2rRpDc8dTEOy7iafEOl/ZQSiE2z9H4jqBHHxrfc3NySua4Wa2Lpra5ZV9OOUSabqO7z0buJQOwE5MCq8AlyEstmE+cVtTBI/SF6qn379s5pxc1m9913n3Xp0sXV5uHn/C/Xg5OCEMpJVe5rxKaX7xGbZumtsBM14+WxM+b2Mm794mpqiq9z8xM17rgennGB6EyKyFM7D8d3HrdiJh9DKV/pbYaRNvEnYlx9pnnYcEhHViibCKVZev9rceL7v54XyiZCjRbSG4qsbCI8WX2CCIiAJyAHVgW2cN111zkn1dVXX+1SeEgpYAPKzUSDBw9277Rw4UJXC4vX7bDDDu7GMK5T3mSTTWzChAl28MEHV/CJLXtpbHr5trFplt6W2WiWv24NjLlCG4cVxZ9xYp111ln2zjvvuCgsoq722GMP5xgn7ZS6eaQW5tXS+EpvdemLcXV5lr5bbHzLzXXqd9Wzk9hsIja9suHq2Wq5d5JNiHF4AvoEEYiTgBxYGZ6bP23r06eP21hSE8O3I4880tWl4eayH/zgBzZt2jR35T2pSb7Nnj3bevfu7TaopBGGbrHphUdsmqU3tBW3PpugdhVOLMaJL774whVmX2+99RxI7Iki8h07djSuNw91wp98ak3ZsPS23MbFOGy6VWx8s8x16nct63ex2URsemXD/+cOpkM22YQYh7QvvbcItAYCcmBlfIqk/Oy7777u38iRI23ZsmWuEPszzzzjirdT/woH1RtvvGHbb7+9DR061BVhpp1yyinWpk0btzFde+21M35iy14Wm16+bWyapbdlNprlr1sT47PPPts233xzN04wHtB8qiBRV9xgSnrDAQcckAVNVV7TGF/prQriRsc1MW4549hsuKm5TjZRfzYhG275M2/qHWJjHJtejWtNWaB+LwIiUE0CcmBloOk3mYMGDXLpP88+++xyERI4tB555BGjbkz37t1t/PjxNmzYMOvcubMr4I7Ti00rNxHm0WLTm9zIi3EYC5FNhOGafNcsjP/85z+7Cx1IL6YoKYVjuQ1x8uTJduKJJ7p6WCussEJ4sQnnWWN9Tnpb9ihkEy3j19Rfx8Y361ynftfUky//+9hsIja9suHm22bWv5RNZCXV/NfFyLj531Z/KQKtj4AcWBme6dKlS92m8q9//avttNNOdscdd9iPfvQj8z+fN2+ebbfddnbXXXfZXnvt5d6RSCxq3HDDGY6sPFtsemETm2bpDW/R9cD4lltucQXczzjjDOvatWt4qIlPaA5f6a3sEYlxZbwqfXVsfJs716nfZbeM2GwiNr2y4ey22NxXyiaaSy7738XIOPu30ytFoPUTkAPLzF1f/8orrzhH01ZbbeWeus9B94McP1uyZImLrHrooYecM2uNNdZoeO03v/lNu/jii23IkCHBrSY2vQCJTbP0BjfjurUJbhs86aSTggOulg1Lb/lHJcZhzTg2vtWc69Tv0m0rNpuITa9sWHNzWs+LzY5j0xt2JtW7i0DrI1DXDixCSH/yk5/YDTfc4KIfZs2aZaeeeqqdeeaZrriyd2KR6jNq1Cjr27ev4ajaddddXVogf7v11lvbjBkzXP2ae+65xzbbbLNgVhKbXkDEpll6w9qwbELjROkAGVufkw3LhtMm+djsWHrDznWx8dW4pnFN41r++7pgG0a9sQi0cgJ17cAikooC69dee62rSTNhwgRXk2ajjTayu+++2z16rrGlxlWnTp1sypQp1qVLF6M+BY4ubhrs0aOHTZ8+3QYPHuxq24SsXxObXvjFpll6w9qwbELjROmcGlufkw3LhtPWhbHZsfSGneti46txTeOaxrX893Wt3MegrycCwQjUlQOr9Gr6k08+2Z577jlXlN03oqgOO+wwo+bED3/4Q1ebhoiro48+2l2d69+D8NSnn37a5s6da/vvv7/tvPPOVX9IsekFQGyapTesDcsmNE6UDoyx9TnZsGw4bXKPzY6lN+xcFxtfjWsa1zSu5b+vq/pGUW8oAnVKoG4cWF988YW1a9fOOaF8o8bE/fffb4899lhD5BRX15IaOHPmTFuwYEHNzCI2vYCKTbP0hjdvMQ7LWHzD8tW4Jr5pBNTvwtqF+Iblq3FNfDWuhbcBfYIIiEA4AnXhwBo+fLi98MILtsoqq7jbAwcOHOicWVdffbXddttthiNr7733bqBMRNaBBx5ol1xyiR133HFfiyryLyw9cavWY4pNL987Ns3S+19rDWXDson/jQahGMuGZcOlc45sQjYhm/i/5Q4qtV77+spU40TYcSI2vlqvhV+vVWt/qPcRARH4L4FW7cB6+OGHbejQoe62QP536tSptmjRIleM/fzzz7d//vOf1qtXL+vdu7ede+651rFjRwflo48+sqOOOsq6d+9uo0ePzs1WYtMLmNg0S294cxbjsIzFNyxfjWvim0ZA/S6sXYhvWL4a18RX41r++7rwVqdPEIH6JNBqHVgffvihjRgxwlZaaSW74oorbOWVVzZuE+SGwU8++cSuv/5669Chg40dO9YmTpxoZ599tqtz5dsWW2xh/fr1szFjxuRiGbHpBUpsmqU3vCmLcVjG4huWr8Y18U0joH4X1i7ENyxfjWviq3HNLO99XXir0yeIQP0SaLUOLG4IJOKqZ8+eLpJq2bJl1rZtWzv99NNdOuGjjz7a8NQPPfRQV+/qtNNOs4MPPtheeukl+/GPf2yXXXaZ7bPPPrlYR2x6gRKbZukNb8piHJax+Iblq3FNfNMIqN+FtQvxDctX45r4alzLf18X3ur0CSJQvwRajQPrqquucpFV3bp1sz59+nztif7nP/+xNm3a2PHHH2+rrbaa/fKXv7SlS5e64u3cJDh+/Hi78sornbNr9uzZNnjwYOM9+ZsQLTa9MIhNs/SGtWHZhMaJ0rExtj4nG5YNp83vsdmx9Iad62Ljq3FN45rGtfz3dSH2inpPERCBdALRO7BmzZplhxxyiEsV/Na3vmXPPfeci7q65ppr7Nvf/rZ5x5UvpLzzzjvbsGHDXI0r/zuPhvciEmvrrbe2zp07B7GZ2PQCITbN0hvWhmUTGidKB8fY+pxsWDacNsHHZsfSG3aui42vxjWNaxrX/huskOe+LshmUW8qAiLQKIHoHVjcIPjQQw+5YuLUuHr99ddtjz32sMMPP9zOOecc59Tyjqr58+fbDjvsYE8++aRtueWWDszChQtt4403/pozK5TdxKYXDrFplt5Q1vu/9xXjsIzFNyxfjWthIouTTy02G5ZNyCZKRx3ZsGxCNqF9UvjViD5BBESgUgLROrCIqFqyZIm7UXDTTTd1Rdm9o+rGG2+0a6+91t08OGTIkAYm/GzSpEn2zDPP2Jw5c1zNKyKuqHnVvn37StlV9PrY9PLlYtMsvWFtWDahcaJ00Iutz8mGZcNpE3dsdiy9Yee62PhqXNO4pnEt/31dRZtAvVgERKCqBKJyYFGbimip1VdfvQHCvvvua2ussYZNmTLFvvzyS5dKSDvooINc0XbqWnXq1Mn97NRTT7UvvvjC1l13XVegHecXzi7+PkSLTS8MYtMsvWFtWDahcaJ0bIytz8mGZcNp83tsdiy9Yee62PhqXNO4pnEt/31diL2i3lMERKByAlE4sKZNm2bnnnuuu0mQqKv+/fvbyJEjrV27dnbLLbfYcccdZ3//+99duiBphCuvvLJNnz7d1bnib6l7xd9tvvnm9vbbb9v2229v48aNsx133LFyYhn+Ija9fKXYNEtvWBuWTWicKB3qYutzsmHZcNp0HZsdS2/YuS42vhrXNK5pXMt/X5dh66eXiIAI5Eig0A4snE6XXHKJuyGQdL/ddtvNnnrqKTvjjDPsvvvus969e9u8efPssMMOsy5dutjUqVOdk4vIK9qGG25oI0aMsFNOOcXeffddGzt2rCvwTnRWiBabXhjEpll6w9qwbELjROnYGFufkw3LhtPm99jsWHrDznWx8dW4pnFN41r++7oQe0W9pwiIQMsJFNqBRdH1gQMH2vDhw61Pnz6uJtM3vvEN23///W2dddax3/72t/bVV1+59MFjjz3W7rzzzgbn1Jtvvml77rmnK0A+YMCAlpPK8A6x6eUrxaZZejMYYgtfIsYtBNjEn4tvWL4a18Q3jYD6XVi7EN+wfDWuia/Gtfz3deGtTp8gAiLQHAKFdmDxhSZOnOgirCiy7h1Y/fr1s0022cTVt6J99tlndt5557kC7aeffrpzYk2ePNmlxZFKSJH3vFpsesU4vGXIJsS4lEBsNhGbXo1r6nNpBGKzY+kNa8ex8dW4FtYeYuQbo+YY+114y9MniIAIVEKg8A6s5Jch2mrFFVe0rl272uDBg23YsGHLfVcitR5++GH75JNP3OtIPaT+Va1abHrhFJtm6Q1v3WIclrH4huWrcU180wio34W1C/ENy1fjmvhqXAtvA/oEERCBYhKIyoEFwgULFliPHj3smWeesY022shR9ZFZ1L9aunSpvfHGG/a9732vEMRj0yvG4c1GNiHGpQRis4nY9GpcU59LIxCbHUtvWDuOja/GtbD2ECPfGDXH2O/CW54+QQREoDEC0TmwSA0kdfD555933+v999+3uXPn2i677NJQvL1Ijzw2vbCLTbP0hrd4MQ7LWHzD8tW4Jr5pBNTvwtqF+Iblq3FNfDWuhbcBfYIIiEDxCETjwPK3Cx5//PG20kor2Q033OBuKKT21VlnnWWjR492aYNFabHphVtsmqU3vLWLcVjG4huWr8Y18U0joH4X1i7ENyxfjWviq3EtvA3oE0RABIpLIBoHFgipqUD6IEXZX375ZeMa5Ouuu8569+5dSMKx6RXj8GYkmxDjUgKx2URsejWuqc+lEYjNjqU3rB3HxlfjWlh7iJFvjJpj7HfhLU+fIAIi0BSBqBxYc+bMcbWt1l57baNgO5FXRW6x6YVlbJqlN3wPEOOwjMU3LF+Na+KbRkD9LqxdiG9YvhrXxFfjWngb0CeIgAgUk0BUDiwQEnE1aNAga9euXTGJlqiKTa8Yhzcr2YQYlxKIzSZi06txTX0ujUBsdiy9Ye04Nr4a18LaQ4x8Y9QcY78Lb3n6BBEQgcYIROfA0uMUAREQAREQAREQAREQAREQAREQAREQARGoLwJyYNXX89a3FQEREAEREAEREAEREAEREAEREAEREIHoCMiBFd0jk2AREAEREAEREAEREAEREAEREAEREAERqC8CcmDV1/PWtxUBERABERABERABERABERABERABERCB6AjIgRXdI5NgERABERABERABERABERABERABERABEagvAnJg1dfz1rcVAREQAREQAREQAREQAREQAREQAREQgegIyIEV3SOTYBEQAREQAREQAREQAREQAREQAREQARGoLwJyYNXX89a3FQEREAEREAEREAEREAEREAEREAEREIHoCMiBFd0jk2AREAEREAEREAEREAEREAEREAEREAERqC8CcmDV1/PWtxUBERABERABERABERABERABERABERCB6AjIgRXdI5NgERABERABERCBohM49thjbeLEiU7mCiusYB07drSuXbvagAEDjN+1adMm01e4+eab7fTTT7ePP/440+v1IhEQAREQAREQARForQTkwGqtT1bfSwREQAREQAREoGYEcFItXrzYJkyYYMuWLXP/PWPGDLvkkktst912s3vvvdc5tppqcmA1RUi/FwEREAEREAERqBcCcmDVy5PW9xQBERABERABEciNAA4soqbuueee5T7zoYcesj333NNuuukmO+GEE+zKK690Tq758+e7KK0DDzzQxo4dax06dLBHHnnEevXqtdzfjxo1yi688EL797//beedd57deuut7nO23npr+8UvfmG77757bt9RHyQCIiACIiACIiACeRKQAytP2vosERABERABERCBuiBQzoHFl+/evbttsMEGdv/999vVV19t3bp1s06dOjkn1tChQ22PPfawcePG2ZdffmnXX3+9XXDBBTZ37lzHDccW/0488UR79dVX7dJLL3Xvdffdd9v5559vs2bNsu985zt1wVhfUgREQAREQAREoL4IyIFVX89b31YEREAEREAERCAHAo05sA4//HB7+eWXnQOqtN1xxx02ZMgQe//9992v0lIIFy5caJ07dzb+F+eVb3vttZfttNNONmbMmBy+oT5CBERABERABERABPIlIAdWvrz1aSIgAiIgAiIgAnVAoDEHVv/+/W327Nn2yiuv2AMPPODqYs2ZM8f+9a9/2dKlS23JkiX2+eef2yqrrJLqwJo2bZodcMABtuqqqy5HkrTCvn372u23314HhPUVRUAEREAEREAE6o2AHFj19sT1fUVABERABERABIITaMyBxW2EG2+8sV177bW25ZZb2sknn2w4taiB9fjjj9ugQYPso48+sjXXXDPVgYWDauDAgc4B1rZt2+W+C+mF66+/fvDvpw8QAREQAREQAREQgbwJyIGVN3F9ngiIgAiIxkBZzQAAAmVJREFUgAiIQKsn0FQR9/Hjx9vqq69uAwYMcBFXbdq0cUxGjx5tI0eObHBg3XLLLXbSSSfZp59+2sDsb3/7m22xxRb26KOPuhsN1URABERABERABESgHgjIgVUPT1nfUQREQAREQAREIFcCOLAWL17sbhhctmyZ++8ZM2a4dEFuCuR2QtIIKehOIXduH3ziiSfsnHPOsUWLFjU4sJ588knr0aOHSzWk2Dtphfw78sgj3euvuOIK23bbbe29996zBx980Iju6t27d67fVR8mAiIgAiIgAiIgAnkQkAMrD8r6DBEQAREQAREQgboigANr4sSJ7juvsMIKttZaazkH1BFHHGHHHHNMQ8TVVVddZZdddpl9/PHH1rNnT5caePTRRzc4sPh7UgynTp1qH3zwgY0aNcouvPBC++qrr1y01qRJk5zDa5111rFddtnFLrroIttmm23qirW+rAiIgAiIgAiIQH0QkAOrPp6zvqUIiIAIiIAIiIAIiIAIiIAIiIAIiIAIREtADqxoH52Ei4AIiIAIiIAIiIAIiIAIiIAIiIAIiEB9EJADqz6es76lCIiACIiACIiACIiACIiACIiACIiACERLQA6saB+dhIuACIiACIiACIiACIiACIiACIiACIhAfRCQA6s+nrO+pQiIgAiIgAiIgAiIgAiIgAiIgAiIgAhES0AOrGgfnYSLgAiIgAiIgAiIgAiIgAiIgAiIgAiIQH0QkAOrPp6zvqUIiIAIiIAIiIAIiIAIiIAIiIAIiIAIREtADqxoH52Ei4AIiIAIiIAIiIAIiIAIiIAIiIAIiEB9EPh/D9xpklROZ+sAAAAASUVORK5CYII=" width="1200">





    <AxesSubplot:xlabel='Date'>



# Time Shift


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>dAsk_open</th>
      <th>dAsk_high</th>
      <th>dAsk_low</th>
      <th>dAsk_close</th>
      <th>dBid_open</th>
      <th>dBid_high</th>
      <th>dBid_low</th>
      <th>dBid_close</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-08-20 14:53:01</th>
      <td>31.250120</td>
      <td>34.059719</td>
      <td>31.250120</td>
      <td>33.092663</td>
      <td>24.058703</td>
      <td>29.033866</td>
      <td>23.386813</td>
      <td>27.527786</td>
    </tr>
    <tr>
      <th>2020-08-20 14:54:01</th>
      <td>33.101373</td>
      <td>35.544810</td>
      <td>33.101373</td>
      <td>33.966106</td>
      <td>27.680442</td>
      <td>29.661571</td>
      <td>25.809777</td>
      <td>28.331556</td>
    </tr>
    <tr>
      <th>2020-08-20 14:55:01</th>
      <td>33.512670</td>
      <td>35.409447</td>
      <td>33.512670</td>
      <td>32.731117</td>
      <td>27.442044</td>
      <td>30.040872</td>
      <td>25.757473</td>
      <td>26.515337</td>
    </tr>
    <tr>
      <th>2020-08-20 14:56:01</th>
      <td>32.953497</td>
      <td>38.374924</td>
      <td>32.953497</td>
      <td>36.938822</td>
      <td>26.960129</td>
      <td>34.502927</td>
      <td>26.182525</td>
      <td>33.170154</td>
    </tr>
    <tr>
      <th>2020-08-20 14:57:01</th>
      <td>35.280327</td>
      <td>40.091153</td>
      <td>35.280327</td>
      <td>38.888639</td>
      <td>32.502927</td>
      <td>36.118225</td>
      <td>30.629955</td>
      <td>34.707211</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>dAsk_open</th>
      <th>dAsk_high</th>
      <th>dAsk_low</th>
      <th>dAsk_close</th>
      <th>dBid_open</th>
      <th>dBid_high</th>
      <th>dBid_low</th>
      <th>dBid_close</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-09-30 15:25:45</th>
      <td>-189.819567</td>
      <td>-186.931951</td>
      <td>-192.449672</td>
      <td>-188.342037</td>
      <td>-196.240383</td>
      <td>-192.915087</td>
      <td>-198.228605</td>
      <td>-194.748284</td>
    </tr>
    <tr>
      <th>2020-09-30 15:26:45</th>
      <td>-187.480936</td>
      <td>-183.198253</td>
      <td>-188.952119</td>
      <td>-186.499092</td>
      <td>-193.948305</td>
      <td>-189.566043</td>
      <td>-194.443868</td>
      <td>-192.466745</td>
    </tr>
    <tr>
      <th>2020-09-30 15:27:45</th>
      <td>-186.657086</td>
      <td>-184.786044</td>
      <td>-188.106640</td>
      <td>-185.362550</td>
      <td>-192.397959</td>
      <td>-190.866729</td>
      <td>-193.666678</td>
      <td>-191.238705</td>
    </tr>
    <tr>
      <th>2020-09-30 15:28:45</th>
      <td>-185.642743</td>
      <td>-185.128182</td>
      <td>-191.813965</td>
      <td>-191.617835</td>
      <td>-191.414513</td>
      <td>-191.134252</td>
      <td>-196.286645</td>
      <td>-196.286645</td>
    </tr>
    <tr>
      <th>2020-09-30 15:29:45</th>
      <td>-191.648428</td>
      <td>-187.964615</td>
      <td>-191.898101</td>
      <td>-189.069005</td>
      <td>-196.309604</td>
      <td>-193.486792</td>
      <td>-197.023329</td>
      <td>-194.575801</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.shift(periods=1).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>dAsk_open</th>
      <th>dAsk_high</th>
      <th>dAsk_low</th>
      <th>dAsk_close</th>
      <th>dBid_open</th>
      <th>dBid_high</th>
      <th>dBid_low</th>
      <th>dBid_close</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-08-20 14:53:01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2020-08-20 14:54:01</th>
      <td>31.250120</td>
      <td>34.059719</td>
      <td>31.250120</td>
      <td>33.092663</td>
      <td>24.058703</td>
      <td>29.033866</td>
      <td>23.386813</td>
      <td>27.527786</td>
    </tr>
    <tr>
      <th>2020-08-20 14:55:01</th>
      <td>33.101373</td>
      <td>35.544810</td>
      <td>33.101373</td>
      <td>33.966106</td>
      <td>27.680442</td>
      <td>29.661571</td>
      <td>25.809777</td>
      <td>28.331556</td>
    </tr>
    <tr>
      <th>2020-08-20 14:56:01</th>
      <td>33.512670</td>
      <td>35.409447</td>
      <td>33.512670</td>
      <td>32.731117</td>
      <td>27.442044</td>
      <td>30.040872</td>
      <td>25.757473</td>
      <td>26.515337</td>
    </tr>
    <tr>
      <th>2020-08-20 14:57:01</th>
      <td>32.953497</td>
      <td>38.374924</td>
      <td>32.953497</td>
      <td>36.938822</td>
      <td>26.960129</td>
      <td>34.502927</td>
      <td>26.182525</td>
      <td>33.170154</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.tshift(freq='3H').head()
```

    <ipython-input-32-5818ed78a47f>:1: FutureWarning: tshift is deprecated and will be removed in a future version. Please use shift instead.
      df.tshift(freq='3H').head()
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>dAsk_open</th>
      <th>dAsk_high</th>
      <th>dAsk_low</th>
      <th>dAsk_close</th>
      <th>dBid_open</th>
      <th>dBid_high</th>
      <th>dBid_low</th>
      <th>dBid_close</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-08-20 17:53:01</th>
      <td>31.250120</td>
      <td>34.059719</td>
      <td>31.250120</td>
      <td>33.092663</td>
      <td>24.058703</td>
      <td>29.033866</td>
      <td>23.386813</td>
      <td>27.527786</td>
    </tr>
    <tr>
      <th>2020-08-20 17:54:01</th>
      <td>33.101373</td>
      <td>35.544810</td>
      <td>33.101373</td>
      <td>33.966106</td>
      <td>27.680442</td>
      <td>29.661571</td>
      <td>25.809777</td>
      <td>28.331556</td>
    </tr>
    <tr>
      <th>2020-08-20 17:55:01</th>
      <td>33.512670</td>
      <td>35.409447</td>
      <td>33.512670</td>
      <td>32.731117</td>
      <td>27.442044</td>
      <td>30.040872</td>
      <td>25.757473</td>
      <td>26.515337</td>
    </tr>
    <tr>
      <th>2020-08-20 17:56:01</th>
      <td>32.953497</td>
      <td>38.374924</td>
      <td>32.953497</td>
      <td>36.938822</td>
      <td>26.960129</td>
      <td>34.502927</td>
      <td>26.182525</td>
      <td>33.170154</td>
    </tr>
    <tr>
      <th>2020-08-20 17:57:01</th>
      <td>35.280327</td>
      <td>40.091153</td>
      <td>35.280327</td>
      <td>38.888639</td>
      <td>32.502927</td>
      <td>36.118225</td>
      <td>30.629955</td>
      <td>34.707211</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['dBid_close'].tail(200).plot(figsize=(16,  6))
df['30 MA'] = df['dBid_close'].rolling(window=30).mean()
df['30 MA'].tail(200).plot()
```




    <AxesSubplot:xlabel='Date'>




![png](output_34_1.png)



```python
df['dBid_close'].expanding().mean().plot(figsize=(16, 6))
df['30 MA'].plot()
```




    <AxesSubplot:xlabel='Date'>




![png](output_35_1.png)


# Bollinger Bands


```python
# Close 20 MA
df['20 MA'] = df['dBid_close'].rolling(20).mean()

# Upper = 20 MA + 2 * std(20)
df['Upper'] = df['20 MA'] + 2 * (df['dBid_close'].rolling(20).std())

# Lower = 20MA - 2 * std(20)
df['Lower'] = df['20 MA'] - 2 * (df['dBid_close'].rolling(20).std())

# Close
df[['dBid_close', '20 MA', 'Upper', 'Lower']].tail(200).plot(figsize=(16, 6))
```




    <AxesSubplot:xlabel='Date'>




![png](output_37_1.png)



```python

```

```python
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib notebook
```


```python
silver_rate = pd.read_csv('silver_rates.csv', index_col = 'Date')
```


```python
silver_rate.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>dAsk_open</th>
      <th>dAsk_high</th>
      <th>dAsk_low</th>
      <th>dAsk_close</th>
      <th>dBid_open</th>
      <th>dBid_high</th>
      <th>dBid_low</th>
      <th>dBid_close</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>08/20/2020 14:53:01</th>
      <td>31.250120</td>
      <td>34.059719</td>
      <td>31.250120</td>
      <td>33.092663</td>
      <td>24.058703</td>
      <td>29.033866</td>
      <td>23.386813</td>
      <td>27.527786</td>
    </tr>
    <tr>
      <th>08/20/2020 14:54:01</th>
      <td>33.101373</td>
      <td>35.544810</td>
      <td>33.101373</td>
      <td>33.966106</td>
      <td>27.680442</td>
      <td>29.661571</td>
      <td>25.809777</td>
      <td>28.331556</td>
    </tr>
    <tr>
      <th>08/20/2020 14:55:01</th>
      <td>33.512670</td>
      <td>35.409447</td>
      <td>33.512670</td>
      <td>32.731117</td>
      <td>27.442044</td>
      <td>30.040872</td>
      <td>25.757473</td>
      <td>26.515337</td>
    </tr>
    <tr>
      <th>08/20/2020 14:56:01</th>
      <td>32.953497</td>
      <td>38.374924</td>
      <td>32.953497</td>
      <td>36.938822</td>
      <td>26.960129</td>
      <td>34.502927</td>
      <td>26.182525</td>
      <td>33.170154</td>
    </tr>
    <tr>
      <th>08/20/2020 14:57:01</th>
      <td>35.280327</td>
      <td>40.091153</td>
      <td>35.280327</td>
      <td>38.888639</td>
      <td>32.502927</td>
      <td>36.118225</td>
      <td>30.629955</td>
      <td>34.707211</td>
    </tr>
  </tbody>
</table>
</div>




```python
silver_rate['dAsk_close'].plot(figsize=(8, 4), c='red', ls='--')
```


    <IPython.core.display.Javascript object>



<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAyAAAAGQCAYAAABWJQQ0AAAgAElEQVR4XuydBbgdxfmHv1KKNRDcEzxYgODursWKS4OH4O4El+AULxDaErS4e9Hyxy1AsUKDFU2hlECB//PeyXDm7N09Z8/ZPefs3vv7noenzdnZmdl35u7ON/PJL3766aefTCICIiACIiACIiACIiACIiACbSDwCykgbaCsJkRABERABERABERABERABLoISAHRRBABERABERABERABERABEWgbASkgbUOthkRABERABERABERABERABKSAaA6IgAiIgAiIgAiIgAiIgAi0jYAUkLahVkMiIAIiIAIiIAIiIAIiIAJSQDQHREAEREAEREAEREAEREAE2kZACkjbUKshERABERABERABERABERABKSCaAyIgAiIgAiIgAiIgAiIgAm0jIAWkbajVkAiIgAiIgAiIgAiIgAiIgBQQzQEREAEREAEREAEREAEREIG2EZAC0jbUakgEREAEREAEREAEREAEREAKiOaACIiACIiACIiACIiACIhA2whIAWkbajUkAiIgAiIgAiIgAiIgAiIgBURzQAREQAREQAREQAREQAREoG0EpIC0DbUaEgEREAEREAEREAEREAERkAKiOSACIiACIiACIiACIiACItA2AlJA2oZaDYmACIiACIiACIiACIiACEgB0RwQAREQAREQAREQAREQARFoGwEpIG1DrYZEQAREQAREQAREQAREQASkgGgOiIAIiIAIiIAIiIAIiIAItI2AFJC2oVZDIiACIiACIiACIiACIiACUkA0B0RABERABERABERABERABNpGQApI21CrIREQAREQAREQAREQAREQASkgmgMiIAIiIAIiIAIiIAIiIAJtIyAFpG2o1ZAIiIAIiIAIiIAIiIAIiIAUEM0BERABERABERABERABERCBthGQAtI21GpIBERABERABERABERABERACojmgAiIgAiIgAiIgAiIgAiIQNsISAFpG2o1JAIiIAIiIAIiIAIiIAIiIAVEc0AEREAEREAEREAEREAERKBtBKSAtA21GhIBERABERABERABERABEZACojkgAiIgAiIgAiIgAiIgAiLQNgJSQNqGWg2JgAiIgAiIgAiIgAiIgAhIAdEcEAEREAEREAEREAEREAERaBsBKSBtQ62GREAEREAEREAEREAEREAEpIBoDoiACIiACIiACIiACIiACLSNgBSQtqFWQyIgAiIgAiIgAiIgAiIgAlJANAdEQAREQAREQAREQAREQATaRkAKSNtQqyEREAEREAEREAEREAEREAEpIJoDIiACIiACIiACIiACIiACbSMgBaRtqNWQCIiACIiACIiACIiACIiAFBDNAREQAREQAREQAREQAREQgbYRkALSNtRqSAREQAREQAREQAREQAREQAqI5oAIiIAIiIAIiIAIiIAIiEDbCEgBaRtqNSQCIiACIiACIiACIiACIiAFRHNABERABERABERABERABESgbQSkgLQNdf2GfvzxR/vggw9s0kkntV/84hf1b1AJERABERABERABERCBthL46aef7KuvvrIZZ5zRxhtvvLa23VMakwJSoJEcPXq09evXr0A9UldEQAREQAREQAREQATiCPzzn/+0mWeeWXCaICAFxMwefvhhGz58uD3zzDP24Ycf2o033mgbbrjhzzjRdI8++mi75JJL7Msvv7Rll13WLrjgAptrrrl+LvP555/bnnvuabfeemuXNrzJJpvY2WefbX369Ek9LGPGjLHJJ5/cmNCTTTZZ6vtUUAREQAREQAREQAREoD0E/v3vf3dtGLMm7Nu3b3sa7WGtSAExszvvvNMee+wxW3TRRW3jjTfupoCccsopdtJJJ9kVV1xhs802mx155JH20ksv2ahRo2yiiSbqmhJrr712l/Jy0UUX2ffff2+DBw+2xRdf3EaOHJl6yjChmcgoIlJAUmNTQREQAREQAREQARFoGwGt17KjlgISYYjvRXgCwukHNn7777+/HXDAAV2lURCmm246GzFihG2xxRb26quv2nzzzWdPPfWULbbYYl1l7rrrLltnnXUMsyruTyOa0GkoqYwIiIAIiIAIiIAIdI6A1mvZ2UsBqaOAvP322zbHHHPYc889Z4MGDfq59Iorrtj1b8ysLrvssi4F5Ysvvvj5+v/+97+u05HrrrvONtpoo1QjpQmdCpMKiYAIiIAIiIAIiEDHCGi9lh29FJA6Csjjjz/e5fNBdKoZZpjh59KbbbZZV6Sqa665xk488cQu86zXX3+9qrZpp53WjjnmGBsyZEjsSI0dO9b4z4u3KZQJVvaJrRpEQAREQAREQAREoBUEpIBkpyoFpIMKyLBhw7oUlKhIAck+sVWDCIiACIiACIiACLSCgBSQ7FSlgNRRQFppgqUTkOwTWDWIgAiIgAiIgAiIQDsJSAHJTlsKSB0FxDuh44COnwfCxMO8KuqE/vTTT3dF0kLuueceW2utteSEnn2OqgYREAEREAEREAERKAwBKSDZh0IKiJl9/fXX9uabb3bRXHjhhe2MM86wlVde2aacckrr37+/EYb35JNPrgrD++KLL3YLw/vxxx/bhRde+HMYXiJiKQxv9kmqGkRABERABERABESgKASkgGQfCSkgZvbQQw91KRxR2X777btOOXwiwosvvrgr6cxyyy1n559/vg0YMODnW0hEuMcee1QlIjznnHMaSkSoCZ19QqsGERABERABERABEWglAa3XstOVApKdYW41aELnhlIViYAIiIAIiIAIiEBLCGi9lh2rFJDsDHOrQRM6N5SqSAS6E/jxR7PxxhMZERABERABEchEQOu1TPi6bpYCkp1hbjVoQueGUhWJQDWBu+8223xzs0suMfvtb0VHBERABERABJomoPVa0+h+vlEKSHaGudWgCZ0bSlUkAtUE5pzT7K23zKad1uzjj0VHBERABERABJomoPVa0+ikgGRHl38NmtD5M1WNItBFYPHFzZ5+2sH46SdBEQEREAEREIGmCWi91jQ6KSDZ0eVfgyZ0/kxVowh0EfjFLyogpIBoUoiACIiACGQgoPVaBnjjbpUJVnaGudWgCZ0bSlUkAtUEpIBoRoiACIiACOREQOu17CClgGRnmFsNmtC5oVRFIlBNYMYZzT780P2mExDNDhEQAREQgQwEtF7LAE8nINnh5V2DJnTeRFWfCJjZDz+YrbOO2T33mC20kNnzzwuLCIiACIiACDRNQOu1ptH9fKNOQLIzzK0GTejcUKoiEagQWGEFs0ceqfybfCChSZZYiYAIiIAIiEADBLReawBWQlEpINkZ5laDJnRuKFVRbyaAqRVKxkwzOQpRZePbb80mnLA3E9Kzi4AIiIAIZCCg9VoGeONulQKSnWFuNWhC54ZSFfVWAv/7n9mvfuWe/ptvzCaeuLsC8sUXZpNP3lsJ6blFQAREQAQyEtB6LSNAZULPDjDPGjSh86Spunolga++MptsMvfo775r1r9/dwXkuefMBg3qlXj00CIgAiIgAtkJaL2WnaFOQLIzzK0GTejcUKqi3krg3/8269vXPf0//mE2yyzdFZCddjK78EKzX/6yt1LSc4uACIiACGQgoPVaBnjjbpUCkp1hbjVoQueGUhX1VgL//a/ZJJO4p3/nHbNZZzU77TSzAw+sJjJkiNn55/dWSnpuERABERCBDAS0XssATwpIdnh516AJnTdR1dcrCXin88GDzS67zOyf/3SmWFFRPpBeOT300CIgAiKQlYDWa1kJmukEJDvD3GrQhM4NpSrqzQSiWc//8hezTTeVAtKb54SeXQREQARyJKD1WnaYUkCyM8ytBk3o3FCqot5K4PvvzSaYoPL0nHKsuKLZww93J3L88Wbrr2+24IK9lZaeWwREQAREoAkCWq81AS1yixSQ7Axzq6FjE/qtt8xGj3YLNYkIlJkAc3nOOasVkOWWM3vsMbP55zd75ZXkk5CxY82+/tqF7vV+JGVmob6LgAiIgAi0hEDH1msteZrOVCoFpDPcY1vt2IT2JisPPGC28soFIqKuiECDBG6+2WzDDasVkGmmMfv0U7MzzjA77zwzlJRQvC/I739vtueeZksvbfb44w02rOIiIAIiIAK9hUDH1ms9CLAUkAINZscmdGgzTxShiSYqEBV1RQQaIBDNeo5y4X/bfnuzfv3MML2KKiD/+pfZdNNVKy4NNKuiIiACIiACvYdAx9ZrPQixFJACDWbHJnS4aPPJ2wrERV0RgdQEtt7abOTISvFvv60o1OT/WH11sxtvNLv66kqZN94wI2LWo49KAUkNWgVFQAREoPcS6Nh6rQchlwJSoMHs2IQOFZC33zabbbYCUVFXRKABAu+955IPevn448rJxtNPmy26qNmXX5pNMUXtSseMqWRUb6B5FRUBERABEej5BDq2XutBaKWAFGgwOzahpYAUaBaoK5kJXHed2WabuWqefNJsySXd///f/1z2cxSRxRev3QwZ1SedNHNXVIEIiIAIiEDPI9Cx9VoPQikFpECD2bEJHc2bUCAm6ooINETgkkvMdtmlcsv44zvFgzn+44/uv6eeMltqqdrVfvWVWZ8+DTWtwiIgAiIgAr2DQMfWaz0IrxSQAg1mxya0FJACzQJ1JROBZZdNjmCFQzoO6EceGd/E5pub4SeywAJmRM4ab7xMXdHNIiACIiACPZNAx9ZrPQinFJCUgzls2DA75phjqkrPPffc9tprr3X99u2339r+++9vV199tY0dO9bWXHNNO//88226MLJOnbY6NqGlgKScBSpWeALRKFhhh1FAjjrK7Ljjkh8Dk60llij8Y6qDIiACIiACnSPQsfVa5x4595algKREigJy/fXX23333ffzHeOPP75NPfXUXf8eMmSI3X777TZixAjr27ev7bHHHjbeeOPZYyRASykdmdBkiA4TEKJQzT13yh6rmAgUjEA9BYQEg4SaTpK99zY766yCPZS6IwIiIAIiUCQCHVmvFQlADn2RApISIgrITTfdZM8//3y3O8aMGWPTTDONjRw50jbddNOu65yMzDvvvPbEE0/YUvXszcfV2JEJHV2w7bOP2ZlnpqSiYiJQMAL1FJBa1/2jDBlidvrpLiO6RAREQAREQAQiBDqyXuthoyAFJOWAooAMHz6863RjookmsqWXXtpOOukk69+/vz3wwAO26qqr2hdffGGTTz75zzXOMsssts8++9i+++4b2wqmWvznhQndr18/Q6GZbLLJUvYsY7G4BZnPDJ2xat0uAm0nUE8BmXdedgfqd0theOszUgkREAER6KUEpIBkH3gpICkZ3nnnnfb1118bfh8ffvhhlz/I+++/by+//LLdeuutNnjw4CplgmqXWGIJW3nlle2UU06JbSXOr4SCUkBSDoqKiUCUwCKLmD33XDwXFOuBA81eeaU+Nykg9RmphAiIgAj0UgJSQLIPvBSQJhl++eWXxgnHGWecYRNPPHFTCkghT0Cmmsrs00+bpKLbRKDDBJJOQP75T7OZZzbbay+zc8+t30kpIPUZqYQIiIAI9FICUkCyD7wUkAwMF198cVtttdVs9dVXb8oEK9p0RyZ03ILtgw9ctmjMVSQiUCYCSQqINyscNswsEs0u9vGY/337lunJ1VcREAEREIE2EejIeq1Nz9auZqSANEkacyz8PzCj2n777buc0K+66irbZJNNump8/fXXbZ555imfE3rIY/Ros5lmapKQbhOBNhPg1CLwwapq3SsgJCWccEKXkJCIV2efHd/JZ54xO+AAs4MPNltzzTY/iJoTAREQAREoMgEpINlHRwpISoYHHHCArb/++l1mVx988IEdffTRXRGxRo0a1aV8EIb3jjvu6ArDiwP5nnvu2VXz448/nrIFs45M6FpOu7fdZrbuuqn7r4Ii0FEC/M39/vfxXQgDK2y0kdlNN7lob5hjvf222aGHmt18s9moUe7+QYPMfMQ7BWXo6LCq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OQmQ0C/q98ZmE4v2//43OV9SO582TgHhFJsNJEKrk/Rv4ECzpZc2I9QtPiPRe9q12Oebhr/hV1/VJ9SuPtXviUpECJR+vVaAEZUCEgzCiBEjupSG6AnInXfeaeutt5598MEHNt1003XdceGFF9rBBx9sn3zySdepCf8fZePlIF76Flts0VXXXXfdlWqoNaFTYWpNofffN5t55u514wgfJkOUAtIa/qpVBESguASii/XPP68E7ShCr+MUEBbvJJQlVxP+hGFERZQqwvKSH8RLuxb7oblvHLs+fcy+/tpdaVefijCGJeuD1mvZB0wKSAoF5KijjrJbbrmly0TLCyces88+uz377LO28MIL2worrGCLLLKInUUm03Fy+eWXdyk0Y8aMSTVSmtARTEQvIbs4H7tJJ3V5ObCLjQovaUJFkoMjzbF23GhgHkAkq333rb5KNnZ2zGoJ+UMwF9MHI9U8VyEREIGSEYgu8IuW8yjav0cfNVt22YoCQh6T8CT7zTed0zlO314IHYw5Gd+QcRuNLRkl2qhVP/30eVekgLRkCPKoVOu17BSlgKRQQHbZZRd799137W6coMfJN998Y7/+9a/tjjvusLXXXtsGDBhggwcPtkMPPfTnMlxbd911jbITxyyMx44da/znhQmN7wgKy2REFSmbkBmc0IbI7LNn733otEht2Bnz8o4KIRYHDzZbZhm329WsXHhhxd/E10F4RuquJTLBapa47hMBESgDAXzk3n7bvWORd98169+/OD0P38G33GK2/vqub3PPbfb3v3fvJ76FO+5Y+b1fPzOiYmHui7Ry4U+m87iNNN8bvnskf1x+ebOHHy4OY/WkioAUkOwToscqIIcccoidkpTJehy3V1991eaZZ56fKSaZYLVKARk2bJgdQzKiiJRWAfF+FNgLo4xkFUIk8lHwgh8ImXWjkpcCwEdpp52qa99/f7PTTqv9JJx6cXKC0pjytCsrGt0vAiIgAm0nwGYP77i992570zUbDL8BJO8jiR+S5DyP6ZVXUijHhhlKgd/A4vsV9XvJ64n99yKpvpNPNiP3CslwOdWXFJKAFJDsw9JjFRB8Mz6rE9UCEyr8N7wkKSCtMsHqcScgrVZAVlnFZbb18tBDZvjc7Lln5bdmd66wucUHJKpAEBGtb9/af2mYaeEngpM8WXclIiACIiAC7SMQKhq3317JpZGkgKy0khnfDy9+I/K119wvRLf0ZrV5PgWKDZHCaokPA4/J90IL5dm66sqRgBSQ7DB7rALSDJp6TuhEv5p22mm7qr744ovtwAMPtH/961824YQTdjmhY3L1EhE3xslWW21ln3/+ee9xQs9bAbn2WrPNN68MJR+NBx+s/JvwvJhNhdKsAvLGG2YDBlRqwoYYc4Na4Sd96T/+0Wz77d2/mm2/mQmre0RABERABMx23tnsD39wJPgm7Lqr+/9p3t+UY/Porbcay//UDHc2uvBnTCP4gnCSU88EOE1dKpM7ASkg2ZFKATHcFt7rUhRwNCfM7iOPPNJFds455+zK+eHD8M4444x26qmn2kcffWTbbrttV7jdaBjeoUOH2g477GAPPPCA7bXXXr0rDG+rFRDC+xLC0At2u6NHV/8V4Eg4ySTxfxmXX+5sfLERjvqovPpq9+NudsGIWBKXVCtsgVCPJLKSApL9jaQaREAERKBRAvhLsGmEhAkI0yogq61mdt991a22YjOJ0LuN+ne2oh+N8lX5bgSkgGSfFFJAzLqSBV5xxRXdaJJ0cCV23bt87t7tSkRIskGcz0lEePLJJ3dLRLjvvvvaqFGjbOaZZ7YjjzyydyUizFsBueYasy22qB6Xww93Md2R6MeFyCIoDETNijOb8uXXWceMY/pQXnjBbNAg9wuKDTa4yDvvmM06a+2/ND54RFRB9LHI/lZSDSIgAiLQCIHwW9CMArLqqtXmva16l5Mrq55Jb/S5SVo44YSN0FDZNhCQApIdshSQ7Axzq6H0E7oRBYQwjuwGzT9/8pF0nAICbSJhsdBPCmWYFDrXf6Qw67r66upxO+MMMxzOo0K4xjnmqD3GfCBQlNZeu3L0n9usUEUiIAIiIAI1CYQKCOHbF13UFU97AoIDOt+vUFqxmYRPIUkIo0KUTE74OcGPSq1TfU2LjhEo/XqtY+QqDUsBKcAg+C6UfkKHCgiZcv/v/9yiPC6aiP8wkCBqo43iR4Esu7yw4zKU43ex3Xbx9yUlD8SJHR+Sq67qfrISRsDaZhuzP//Z1d2Kj1CB5py6IgIiIAKlJxAqGpxmL7hgYwoIpTnp/sc/Kigw90UpyFMeeMCM0xaE795hh5ktvrhzTP/1r933Lips1JGcUFIoAqVfrxWAphSQAgxCj1FAiCB1xBHupU3OlNdfN0vKo+E/GDgPkgCwlsTtYs00kxnZy+MEHx4y4EZlxRVdXHWc23/7W+e0SD283Anxi6y3nhmKyn77mWEXfO+9BZoh6ooIiIAIiEA3AuE34rnnKua00W8HptY+YEg9jOToCqJk1iue6vqdd1YidGHqi0Iy11y1b9UJSCq07S4kBSQ7cSkg2RnmVkOPmtBrreWUEF74cScV/sNw4olmQfLGWJhxCghmUUQtiZNPPzWbaqruV1BKiPNOllnCLmL+FZUTTnAmYfh0kIMEMzCJCIiACIhAcQmE34ibbjL7zW9cX8PfydVEwJB6QUX8U+a58KcuckxFTX+/+cYFTcFKIAywEpLWKXwh512PWq91iLAUkA6Bj2u2R01owgfecUf9ExBAkMfjnHO6I3nmGafA4FQYFRSMpDwvBx9sRjKnqPiP0ZprmlGGk45QuP7jj04h4n52puKy6BZozqgrIiACItDrCYSKBgFEyOmEkOh32DD3/4mAiBISmlVtuaUzyY0KIXD5vV7OjrTgaZcEhKH4qI44pqOAJJlZSQFJS7mt5XrUeq2t5CqNSQHpEPi4Zks/odnBIQoVsssuZmSbveSS7tnFuR491Yh7yfIB2Gqr+BHiZU1M9SSJq8+fyhBJi1OSiy5yd6OQcFrj+8rvu+3mHBlxaJSIgAiIgAgUl0D4PQnf/ZgA77ij6/cBB5gNH1759hB2l8S2J50U/1x5Lvzxhbzrrup28DkhyqKXJIf5PPvRyAjiezLRRPkpYY20XYKypV+vFYCxFJACDILvQukn9EcfmRFNhJ0dThIQ7G1HjOhOOY0CMnKk2dZb1x6huOgl3IEyRBu+Hex5eZki0ePuJZZwDvMIL3v6Tkx5wvI2GrO9QPNJXREBERCBXkEgSQH5+GOz6ad3CPBPPO44Z4b74otuk4mTbhzB4yRp4U+Y97PPdqbF9SIk+npJjkji3FDowwILVH5JUkD4riZFfGzV4LK5R0Z2Mre3IiN8q/rdxnpLv15rI6ukpqSAFGAQfBdKP6G9AhIyjcu5wfW8FJBtt3UnLDiYh8KR95lnul+49sQTZt99V3+0Oa7HTlgiAiIgAiJQDgJJCgi9Z5H/8stmJJvF9y8UFJAkH0SCmcw7r9mUU1Z/r+ae25nm4rsRFzY3jtj555sNHVp9JargoBD5U/mwJBEl2Whrp/DsPjBLK5zx2/ksLWqr9Ou1FnFppFopII3QanHZ0k/oOAUE86bo0TMco0fScbtNV15pRkjcWsKHBadCb/PbzBjNOacZ+T6QQw5JPpJvpm7dIwIiIAIi0FoCmM9i9otEvyUsoPGzYEc/KrUUEF+WewlM4qWWspP0lPgcnnpq9dVoP59/3mzhhbvXMHq0i9YYCt89TikGDHAKUi3BJ4ZToEb8WcjTNXCgqzX6/K0dydLUXvr1WgFISwEpwCBU3nP/tr59+9qYMWNssjKa/sQpICuv7EINRiWMh861uIgjXgEhHC7H1byECbEYFdr1x+xZxzMpglbWenW/CIiACIhA6whgNkvgkEbMlW65pRIxK6lnJL4NlRc2u3wI+LS5QqIn/mus4fwOQ+GEZr75uvfivffMCNkbShjRceONk5libkZZTjNQ0jA3wwcyTtEJayGSmM/PRW6SRrO3t26UC1OzFJDsQyEFJDvD3Goo/YSOU0Cuu85s002rGfHSJnP49ddXfo9LSEgyQEysVl/dlcWUil2iqPAxiO4QNTMqHH/7XbRm7tc9IiACIiAC5SFwzz0uCEktefdds/79XQm+NeFpexhxq1YdUQUEv4rxx6++A1OruO8YyRFnmaW67NRTuyiQYcjhuPb5fvqkupiToeT4aI+1+nveeWZ77OFKfPFF+tDF5Rn5zD0t/XotM4HsFUgByc4wtxpKP6HjFJC4rOTffms28cTV3B5/3O3KeEdxrnoFhN0iMp/HnXKg3Bx7bPzOUaMjo52eRompvAiIgAh0ngCJZdmk2nzz7rk2avWORLN8X2rJ3/5mdvvtLrfUPvtUlzzwwO6mVXF1EUyFoCpekhzc4xzRyXc1++zVtfpySy3l/BtDwXEc/0cc8FGQSLyLoPBwDakXWSsMG4yiU8/Mq/MzoO09KP16re3EujcoBaQAg+C7UPoJHaeAxJ1skHzp17+uJr/88mY4vpGoiY8IQhhAjr9RVlBM4pIL8nIklOFii2UfyXov5ewtqAYREAEREIG8CTTjl0EfcCbHqTwUzJ1YuHs56KDaSkaa7wYL/9AHoxEF5I03zPBTDMU/LycjnJCEEp5e+N8x6UaJwYIAc6y//rX2CIQ85QMSy6r067W8/wabqE8KSBPQWnVL6Sf0mDEu0dPll1cQcQTM6UUoOM+FTn1RoHEvZ16CUTvU0GH89793L1gSIDYraT4kzdat+0RABERABFpDoFkFhDxPiy9e3SdORDDNSitpvhuUCRMgplFAUBTYfPvDH7oHWQmfN5qvaoopzDjNT5L99zc77bT0CkhSX4kqOcEEaSn1uHKlX68VYESkgBRgEHwXesyEZlcGxzkkTgHhZMM72VOWExGfwJB74l54cacmfCTwDwklKZZ6vXFOcpavd5+ui4AIiIAIdJZAswoI+Z+WXLK674SOv+OOdM/DqTyBS+oJoeIvvbRSKo0CUkuxqRXGPimcr289yWyMaGH4oeDvEvqnxPXDZ5jHyX2ZZeo9fY+83mPWax0cHSkgHYQfbbrHTGiikGA6hey9t9lZZ1U/KiclhM5FSOYUPSHxL7xnn3W7P+xQYUM74YTV9ZAQit2eUKJOgrfdZrbeeq4EzoOENIwT/E3qJT0s0FxRV0RABERABMYRaFYBwT/Cm/x6mJw6/Pe/6dASTj6NskISQpIRIpwasNjcjLCmAAAgAElEQVSPk6TneOopt5lHJvf116+dRwufyFq5rJI22wjy8vDDzp9y1CjXOxQNEjiGpzf87vsZPX1JR61HlOox67UOjoYUkA7CjzZd+glNBnHC6YYhhNmNueCC6kclqoZ3avvNb8xuvrlynZC7OAYiq65aCeFL5CwymIcSlyAprJuyKDNpTkXIhv7kkwWaDeqKCIiACIhAKgLNKiCYInEi0KxwAp/GXIsTBb5hCObHmBTHSdxzRH064kyfwlOKq64y22qr5CeKU0DwCVlppfh72NSbccbqa76fbA5yitQLpfTrtQKMmRSQAgyC70LpJzRRN+IiVUWPcDm5iHMoB8SHHzqlg2Px0PEuTpFA4YkqFy+8YDZoUGVUuQ9bWhzc60kaW956dei6CIiACIhAewk0q4AQep1NsiyS5rsR9u/008322y++RcyfvAM8zvEs/vGZDIX78eMIhQzvhNhFsDo455zkJyIvCCcd/lSDb26tTOtxiRCb5Z2Fc8HuLf16rQA8pYAUYBB6nQLCDg52tvffn55+nAIS9+IPd3LwG+E4nSPkE06o31aaD0n9WlRCBERABESgnQSaXRBHQ8JjEkwkRr5RUTn3XLM99+z+e73vBlGqZputct9ee5mdfXY8nfnnr5g/JZkML7SQGRttUfEbcksvbUbo4CThtH/ZZV2o3jQSlwgx5I2517BhaWrqUWWkgGQfTikg2RnmVkPpJ3TaExCI8bIkfjm7MUly331mmGRhh3rUUdWnHUm2p2RM5wWN+A8Dx8REO6kn9T4k9e7XdREQAREQgfYTIJwuJwezzmo2xxyNt+8X1CgYKBpRYZHNiQQnFPhg4MfINyz8ziS1OmCAGaF0vdRSQPB53Hln1wbfMpIgRoV+vP56998J5IL5ci3zq7g++gSF9aj57yN9w7cyFL/ZV6+OHnS99Ou1AoyFFJACDILvQukndCMKiH/oev4Z2M6SNRYJy+JrMskk8aOHskJG2V137X5f9A7MsziODtsp0JxQV0RABERABFpMwH9bhg41w+cilCuuMCOKlf8Obbyx84kg9Dvy5pu1lR6CoJDI0Mt115mRQDdOUHRwIp92WpcrixxXUcE3Ms56gGS+zUSkwg/mgAPqA+ZUiFwmcd9sTmQWXLB+HT2oROnXawUYCykgBRgE34XST+gkBYQkTKEDObslZEjnty23rD8CPtqVf/GhXCRFs4qrLUnJIfP6Lbe4ExZ2pRZYoH5fVEIEREAERKBnEfDfiIMPNjvllMqz8Z3acMPui24iN2Ku5aXW6TnfFk5V+NZttJFTPpK+STfd5Mqg8Dz0kFNuooJCEz2BoMxrr5nhC9Ko/OlP7plffrn2nZirEYkyru9xAWEa7UfJypd+vVYA3lJACjAIPV4Bib6cfKhcdlP8rlK92OuhDwiO7jjOpRUUHX9cHt5DxCwfDjhtXSonAiIgAiJQLAKDB5uNGGG2ww7V+TbS9tIvqk880eywwyp3cfqBorHmmtXRrjiB4CQijQKCWZc/LbnrLldXLeHbxjcOZQLTsqigEKGoROWyy9zzNyo8+8CBZi+9VPtOzMEwQYsqIL00h5YUkEYnWvfyUkCyM8ythtJP6KQTEGKqTzRRhRO2urzIiIfOcTORSIhShbM4H5E4CRUQjoyjUUBqjQK+JJhjXXyxCzX41ltm2OVKREAEREAEyk+gWSd0/+T+fr4/nIx7P8LDDzc7/ngXlTEMN0uwE/JmIJwKcDqQJKFTOCcNBx2Ujjf5OHxkq/CO0EwMsyd8RbLIWmuZoRhFBX8aHOi9fPSRGTm+ogoIJzt8u+lvL5LSr9cKMFZSQAowCL4LpZ/QOOax0L/mmmqqzz1XHRqXnRRebigl0YRPxx3nHM5DmWYal9iQ6CQ42ZHQidMTiQiIgAiIgAjkpYCQ8G+xxSqL7JNPNsMsi9OFHXescH700UoAFU4kMNVKkrBvJBJM429BXSQeJEFu3kLYYZ8UkboJ0sJzh3LnnS5XiU/SeMYZZrvs4vxS4kyw2OQLT4Ty7nMB6yv9eq0ATKWAFGAQeowC4h9k9tmrnecWWcTsmWcqpHGsowxO5DiThxL3ciNZYdyRc4HGTl0RAREQARHoEIGsCsgss5gRbpZTDhbkRGd87DF3AsA1THhDP8YbbjDDGd1LLR+QtDlA4tCFUR3zQktiYDKzxwmWCJz2sMHnNwPZEMRP0kvcN5ps8GSF70UiBST7YEsBMVJEnGC33367Pf/88zbBBBPYl19+2Y3sL2L+6K666irbYostfi770EMP2X777WevvPKK9evXz4444gj73e9+l3qUesyEjrKCEdlZvbz9tosa0qeP2VdfVX4n4RJZYqNCDo/QLjc1URUUAREQARHo8QTYnb/kEpdUkAV2o+K/WTffbLbBBmYETuF7FPoIht81HMTDzOFpFRByb+yzT7rePfusGc7uSTlD0tVSXQqzZ/pA9nZOOB58sPq6V8D41T9vNOR9nAIS/cY307eS3dNj1msd5C4FxMyOPvpom3zyyW306NF26aWXJiogl19+ua2FveQ44Z6Jxvk2vPPOOzZw4EDbbbfdbKeddrL777/f9tlnny7FZs16Tmfj6iv9hOYlzH/hThHPRgSQPfaoTHMie8w1l1M2eAl64aUfZ1rVSxMddfC9oKZFQAREoDwECGaCGRGnF82Y5959t7sfn4+kCFXh7ygHnOx7SauAkKE8LplhHOl6Ieq5B58MfC/zkltvNSPKVqiA8P9RuEhgSGJf2sQkOio4sePM3kuk9Ou1AoyTFJBgEEaMGNGlNCSdgNx44422IfaeMXLwwQd3KRsvB6HsOB2hrrviHLxi6ij9hOalxMspKuefX33kS2QPkin17WsWPW2Ke+lir+pzehTgj0ZdEAEREAER6GUEJpuscmLPZhm5o7zgK4GPYpyE3zSiYeFEnkbSKCAnneSCuITO4tTNiRBBV7xgUobfSj0JA8ZE2yd4C6bTtfrVi5L5ln69Vm8utOG6FJAAcj0FZMYZZ7SxY8fa7LPP3nXSMXjwYPOmWSussIItssgidtZZZ/1cIycmKDRjcM6OEeriPy9MaEy3KD8ZL7uySZICwouQ7K6VB3VxzHmBb7ZZ9VPGvdw++MBshhnKRkP9FQEREAER6CkEUDB83ioW2jioH3po5eniFt/RU/3oZlwtNmkUEKJqEYELK4FQ+N5OMYXZqae6X/FhGW+82iMx22xmmEd7Yd1CqHoibWEqzcYhlgtSQLoISQHJ/octBSRgWEsBOe6442yVVVaxSSaZxO65554us61TTz3V9iLJkBHVdUCXQnJo8EK64447bN1117VvvvnGJuboMiLDhg2zY0LnrnHXe5wCwtHtk0+mm63EQCfMLlE33F95vF9IutpUSgREQAREQATyIYBj9rLLmq2yipmP5uhrjlNAWMBPOWWl7Ua+Z2y6Efo2SfCNJO8HVgehckEUL05ACP/rk+tGT2zi6tx++/gw+F7hwEEdn5haCXt/+KG+opPPSHS8Fikg2YegxyoghxxyiJ0SZjSNYfXqq6/aPEHm0FoKSPT2o446yjjh+Cc5LZpUQHrNCUijyaE4TmY3BlEErOx/5apBBERABEQgXwLRE/84BeSTT8ymnba5bxmKzuOP1+4zoXpxVEexIXTuuuu64C4ISgfBXjB1fuGFeMWAnCT+lGS//cxOP717e2lOYvxdRLsMfWPyJV6o2qSAZB+OHquAfPLJJ/bZZ5/VJIQpFVGvvDSigODvsd5669m3335rE044oTVjghXtXOkndJIJFg7oOKJ7we/jgQfci3KNNbqPUZqdpexzXzWIgAiIgAiIQPMEwmSBaUywiDoVRs+q1XKahT9RNi+/PLkWlBBOR/iPsMHRfCX0mazrOLJjDh6sh36uNE0/fGFCFy+zTPM8S3Rn6ddrBWDdYxWQZtg2ooAQuvf000+3zz//vKspnNAxuXqJSBDjZKuttuq63iud0Nlx8dlkBw0yIxmhFyKIENpvppnMRo/uPlTEYyf2upde5NjWzLzVPSIgAiIgAh0gQJJCf4KQ9J0KF/B/+Ut1/pCsCki9LOxh/fQv6geS5tvKyQihe9MIJlqYnfUCkQKSfZClgBj5h97rUhRuueUWGz58uD2CraOZzTnnnNanTx+79dZb7eOPP7alllqqK+zuvffeawcccEDXf96Hw4fhHTp0qO2www72wAMPdPmH9KowvOEJCC+28MWLQxuJB2HLNbKmkg2dpIRRCRWQ55+vKDLZ57tqEAEREAEREIF8CHz3nVNACM+P70WchN9BEupiVpxG0p48pFEifHsEfbnuOvcvHNPTtlGrHCcrJGZE8FshaEwvECkg2QdZCohZV7LAK664ohvNBx980FZaaaWuEwycy99880376aefuhSTIUOG2M4772zjBTsKJCLcd999bdSoUTbzzDPbkUce2bsSEWJahU0qQjzx8KWFLey997oQhOySoHgQUYPIGlHxmdL5nRMmonlIREAEREAERKBoBI44wiUtDCJg/tzF0AeEH2+5xWz99dM9Qfj9xKkci4K4EL6NKCBPPOFMwA480Oz449P1g03DFVZILnvIIWb/+Y8zs+b/Exq4F4gUkOyDLAUkO8PcaugxE5qXLCZYRx1VYcOxLJGwCBdItlVemvPOazZqVHd+hBWceWanwBAFJJrYMDfiqkgEREAEREAEmiTAt8pHuHz/fbMZZ6yuCJNswth6aVYB8UpG9CTi/vtdRK5GBF8PTLfSSr1TEjYb+baj0KAgkeukF0iPWa91cKykgHQQfrTpHjOhBw/uHs6PKB04nP/tb5XH5sWMohInnKageJAtXSICIiACIiACRSPABpnPUUXQmzDkLn31/o6+32Gm8XrPwqnDOHPwrg075MQTXbZ2IlsRLRJrglZLkgLCs2EuvcEGZsOH4whrtuaaZikTL7e6262uv8es11oNqkb9UkA6CL/HKiBE5sCkjZfR3Xe7x8T2NZpFHkf0p58u0AioKyIgAiIgAiKQkkDo98j/n2aaiunxQw+ZTTSR2VJLVSq75x6z1VdPV/lOO5ldeqkrG5pZJZ2GpKu18VJJCkjYpyFDzC68sFI3pmdTT914WyW6QwpI9sGSApKdYW41lH5C89Lp18+F80NIyujtQUksSDSNUHgxY5MqEQEREAEREIGyEfj0U6d0IJyGkBTQL9j32cdsk03Mll/eXZ9sMucDsuKK6Z6SBT5l8asks3mnJE4B4TvP6YeXddZxeUhCacQ3pVPPlqHd0q/XMjx7XrdKAcmLZA71lH5CRx3uSIp0++2ODC/RP/2pQmm33ZxJ1kYb5UBOVYiACIiACIhAmwmQ+Xv88V2jLMhZmPsF+1VXOYUEHw0CrhDRkRORaCjcNne54eZ4Pp4zFBSr66+v/BIqYv7X77+vsElqFEsJ/ENR1kompV+vFYC3FJACDILvQukndFQB6d+/skuCzSqheEMhtngJXzwFmjLqigiIgAiIQCcJeIXjn/+sBE+hPyefbLbwws4U2Qt5r8h/VSbZfHOza68143+vucb13GdgD58jelJCdLDjjqv9pP4eIl8Slr9EUvr1WgFYSwEpwCD0WAVkySVddAyEl1E07N+vfmVGHHWJCIiACIiACJSRAKcamB2/+64Zm25+Uc33jiS8661XeSocx8Mku2V43jgTrDgfj7hytcywyEPiI1wSjCaMFlYCLlJAsg+SFJDsDHOrofQTOnoCQrJBwgTWkh5uJ5rb5FBFIiACIiACxSNAgt3//tfltnrzzYqT+THHmO21l9lrr5ktvbTrdyNRsIrypGkVi3rlyJXSp0/lqTDRmmAC92/CFQ8cWJQnTtWP0q/XUj1lawtJAWkt34ZqL/2EDhWQxx83e+utSmLCOBLbb989XG9DxFRYBERABERABDpIgEU1ifj43s0xR6UjnIZwKoL4xfkf/1j7m9jBx0hsGkuF//2v+nLcxmGcAoKFA/f7a1EFjBwq5FLxp0dFfP6EPpV+vVYA1lJACjAIvguln9ChAsIL6rbbamd9veMOs7XXLtAIqCsiIAIiIAIi0AAB/BvJc/X3v7toVX/9a+XmaMhcnK63266BygtQtN7Jhu/ibLO53CRRQTn79a+7M+EXfv/mG3d6JB+QAgx2e7sgBaS9vGu2VnoFhOSBG2/snhHTKxzWttwy+ZnJiL744gUaAXVFBERABERABBogQPJBIjlhakVQlTAR39VXm22zjYt8xQL7uefMMNkqk6RVQPj+jxrlTKlQyuKEaGAoal5IqsjpykEHmU01VZmoWOnXawWgLQWkAIPgu9BjJjQnG7yMyfOx7LLxhMkLsu++BaKvroiACIiACIhAgwQIJ//VV2aE3eUUIElwuk5K6tdgk20tTnQvFKsbbnDNHnJIJb9XXEeuu85ss83iu/jxx2bTTuuuERXTZ3JHeUlSWtr6sOkb6zHrtfSPnHtJKSC5I22+wh4zobfe2mzkSDPC7BJ28PTTu0NREsLmJ4ruFAEREAERKB6BWgpG2QOuDBhg9sYbLuP5rrsms99qK6eMxQnmVvh9IC+/bLbAAu7/H3us2ZFHFm88a/Sox6zXOkhdCkgH4Ueb7jET2r+AUECI/oGyESdlfyEXaO6oKyIgAiIgAh0m0JMVkNVXN7vvPrM//9mMTcYkqcWAkxTWBZisYbrmhYiZ1F0i6THrtQ4ylwLSQfg9TgH57DMzdkk+/9w92llnOSf0MDJI+NAlPHYt0HRRV0RABERABIpEoCcrIHff7RILr7yy2ZxzNqeA7L23WxdghkXQGi8bbmh2441FGsm6fZECUhdR3QJSQOoial+B0k/oTz81m2aaCjBeNLxwwpfyfPM5RzWEuOBhdIz2oVZLIiACIiACIpCdwDLLuIU5IWZZnOPbECe95cS/lhKGRQQh+qNl5pnH7NVXs49FG2so/XqtjaySmpICUoBB8F0o/YSOKiBnn+0SMS26qNmzz5qdf77Zb35jNtNM7pF/+MFFB5GIgAiIgAiIQBkJzD67CyPLwvq00yrO2tFn6Y0KyHnnmQ0dWk3imWfcmiAUb5pVovEv/Xrt/9s7F3Abq/yPf11C5XZCSS6RayT+zZQotzxGqKmmkkJSKmNMHl3QIQy5NjGNXKamUETKEDWaESVpGlLuaiTkflzLNZX/813be84+2z7nvOe87z57vXt/1/N4HGev97d+6/Nb51jfd63fWhawlgCxIAgJL0B4yRDfeHDZtVix0JnpFB7ht6JaFAe5IgIiIAIiIAKuCPBoWd6AvnQp0LRp1o8kiwBp1izEolcvgC8heSANjyJ2yqhRQN++mTkxAZ2J6AEqEiDegyUB4p2hbxYCP6AjV0BeeCH0S0hFBERABERABBKRALcPffUVMGAAMGxY9B7ec0/WJ0MlGhNuQVuyBGjTJvTCkYUCw2HTqtW5CecUJCNHeiPBNhmLSy/1Zsfl04Gfr7nsZyyrSYDEkm4ubQd+QIcLkH//G6hVC6hUKZcUVF0EREAEREAEAkKgbt2MvMasXH7mGWDIkIB0KAZusv9Dh2ZtmBcRcmUkr2X69IxVlnxaaQr8fC2vrH18TgLER5heTQV+QIcLEL4FKVnSKxI9LwIiIAIiIAL2EuBdFrzTIrvC1ZHsJuD29s4fz6ZOBbp2zdrW44+H8mfyWpw8Uz4vAZJXivn+nARIviPPusHACxAeq5uSEurgxx8DN9xgEV25IgIiIAIiIAI+E2jQAFi9Onuj3HbEXQHJWpxLDLPr/9atwLRpQP/+QOHCuSPFY3znzQs9IwGSO3ZxrC0BEkf4kU0HXoCwQ87xeh06hPZ9cnlaRQREQAREQAQSkcDddwNz5wKnT2fdu0GDgMGDE7H37vrESwd5+aDb4lZEsN7ChUDbthmW3T7r1pcs6iXEfM0jA6+PS4B4Jejj8wkxoMPP9+YRfL//vY+EZEoEREAEREAELCOQ3d0XdPWXX869+8KyLsTUHSdR320jbkXEe+8B7dplWE1NzfogALdtu6yXEPM1l32NVTUJkFiRzYPdhBjQEiB5iLweEQEREAERCCyB7AQItxZVqRLYrvniOEUCxYLb4laA8AXnxIkZVk+eBIoWdduKp3oJMV/zRMD7wxIg3hn6ZiHwA/rgQaBMmQwevHiwRw/f+MiQCIiACIiACFhHIDsB4nYybV2nfHToiiuALVvcGeTRvSdOuKsbyd1rMru7Vk2twM/XctHXWFVNegGydetWDB06FIsXL8aePXtQoUIFdOrUCampqShSpEg69zVr1qBnz55YsWIFypUrh169euEpHh0XVmbPno2BAweCNmvUqIFRo0ahbfjexByiGPgBHXkPiARIrH5uZVcEREAERMAGAl26AK+9luFJw4bAF19k/FsCBChUKLQNzSnnnZd1zszf/gZ07+4uspEC5MorgfXr3T3rsVbg52se++/H40kvQBYuXIhZs2ahY8eOqF69OtatW4fu3bujc+fOeO7ssXAcaDVr1kSrVq3Qv39/rF27Ft26dcO4cePw8MMPmzgsX74cTZs2xYgRI9C+fXvMmDHDCJBVq1ahXr16rmIV+AEdKUC4NProo676rkoiIAIiIAIiEDgCLVoAH36Y4TbzHTZtCv27YkXgu+8C1yXfHY4UCseOARdeGL0ZnmZ1663uXIi28pRPgi/w8zV3hGNaK+kFSDS6Y8aMwcSJE7Hl7JIhv+aKCFdInFWRfv36Ye7cudh09hdNhw4dcOzYMSxYsCDdZKNGjdCgQQNMmjTJVRADP6AlQFzFWZVEQAREQAQShACP2P3gg4zO9OkDdOwIPPts6G4Lbj9K9hIpFE6dyjpXg6tHPNrYTZEAcUPJ2joSIFFCM2DAAHBlZOXKlebTLl26mP1+FBxOWbJkCVq2bImDBw8iJSUFlStXRp8+fdC7d+/0OoMGDTLPrM7ijPBTp06Bf5zCNipVqoQjR46gZBAv8YsUIBRejzxi7eCXYyIgAiIgAiLgiUDr1qE7Pvh/XYkSQL9+mXMhPRlPkIfJ5ejRjM5w3sPVobS0cztI0cZcDjdFAsQNJWvrSIBEhGbz5s245pprzPYrbsViad26NapWrYrJkyen196wYQPq1q0L/l2nTh2zMjJ16lSzlcspEyZMwJAhQ7B3796oA2Dw4MHm88iSEAKEW6/69gUuv9zawS/HREAEREAERMATgZtvDt1F4RTOE85uzfZkN5EevuQSYN++jB799BNAEcK7QR58EHj//YzPGjUCPv3UXe8rVz53i5u2YLljZ0GthBUg3CLFHIzsysaNG1Gb+zXPlp07d6JZs2Zo3rw5Xn755fTvx0qAJPQKyJw5wO23WzDE5YIIiIAIiIAIxIhA+/bAu+9mGB8/HujZM0aNBdRs48aZRUW4SIhcxbj+eibVuuvoZZcBu3Zl1OXFx+vWuXvWY63Ab5n32H8/Hk9YAZKWloYDBw5ky6hatWrpOR27du0ywoN5G1OmTEHBggXTn43VFqxI5wI/oPk2gzeesvB22Fmz/BijsiECIiACIiACdhLgi7aw7dn4+9+Bbt3s9DVeXvFS4j/8IdR6SgrAI/udEilArrsO+M9/3Hka/uxjjwFjx+bbhY+Bn6+5IxzTWgkrQHJDjSsfLVq0MFuvXn/9dRTikXFhxUlC51aq83h8HICnn34ac+bMyZSEfvz4ccyfPz/9ycaNG6N+/frJk4TOnof/Qvj6a6BGjdyEQnVFQAREQAREIDgEIi/Dmz0buPPO4PifX55u3w5cdRVQvHho25TzkjdSgPzqV8CKFTl7tWQJ0LJlRr177wWmT8/5OZ9qSIB4B5n0AoTigysfVapUMTkc4eKjfPnyhjBzMmrVqmVyQfr27WuO6uUxvGPHjs10DC+3b40cORLt2rXDzJkzMXz48OQ6hjdSgLz0EvDQQ95HqSyIgAiIgAiIgI0EeGkeX7Tt3Bny7scfgbMvKm10N24+he+QOH0aKFw45EqkAOE9KqtWRXczNRU4fhx4/vkMAePUZC5Obm5b9whCAsQjQIb+zJl8ytjx7mtMLHC71QMPPBDVdjia8IsIy5Ytay4ipBgJL7yIkCdoORcRjh49OrkuIpQAickYlVEREAEREAFLCfC43QEDQs7xPpBmzSx1NM5uuRUgXCVZs+ZcZylanMuhN24E6tTJXOfLL4Grr863TkqAeEed9ALEO0L/LAR+QB8+HNrf6RQm8vOECxUREAEREAERSEQC99yTke/48cfADTckYi+994k5uWXLhuyEr4D07w+MHJlhP6vbzLny4Vxe+NFHmYXezz+fuyLi3eNsLQR+vhZjPm7MS4C4oZRPdQI/oMN/wZCZkvHyaeSoGREQAREQgbgQCN9CtH49wAm0yrkEeMHg//3fuQJk8WLgppsy6vPrRYvOfZ7H9hYrFp1sHDbyBH6+ZsEYlQCxIAiOC4Ef0BIgFo0muSICIiACIhBzAuECJA4T4Zj3z68GuEWK+R0svAfEOewnUlgwmXzq1IwckfD2o937wc/jwD3w8zW/4urBjgSIB3h+Pxr4AR0pQF55Bcgiv8ZvdrInAiIgAiIgAvlOQALEHXLmdTg5GuEChE9HJqJ//33oVvnIcsUVwJYt535fAsRdDCyrJQFiUUASToC8+irQtatFhOWKCIiACIiACPhIQALEHcy1a4H69UN1cxIgu3cDZ08hTTfOPA/OJ5YtA7ZuzWiTLzn5sjOfS+Dna/nMK1pzEiAWBMFxIfADOnIF5JtvgGrVLCIsV0RABERABETARwLhAoSnM9Wu7aPxBDL17bcZ84HIpPHIFZA33wTuuitz53fsACpVOhcIj+x1tnblI67Az9fykVVWTUmAWBCEhBUgcVgWtSicckUEREAERCDRCYRPnvl2vkmTRO9x3vq3d29oVYO8fvkls41IAcJPebv8b3+bUY+rHlWrntt2nOYZEiB5G8bCUi4AACAASURBVAbhT0mAeGfom4XAD+jwc75JJU6/GHwLiAyJgAiIgAiIQHYEwifPe/YAl1wiXtEI7NuXwSZybjBvHjB5MvDPf2Y82aVLKBn97beBTZtCqydMUI8scZpnBH6+ZsEolQCxIAiOCwkxoMN/GW/bBvDUChUREAEREAERSEQC4f/n8S6sUqUSsZfe+8TE8m7dQisg3GIVbdUj/Hu33w7MmQM0bQrwfhUewXvy5Ll+cDUlmi3vHmdrISHmazFmlJN5CZCcCOXj5wkxoMN/EUybBnTunI8E1ZQIiIAIiIAI5CMBXrjbvXuowaNHMy7Ly0cXEqap8PkDc0CyEirhHZYACWz4JUAsCp0EiEXBkCsiIAIiIAIikBOB4sWBY8dCtfiGvmjRnJ7Q51kRCBcgBQsCTFbPbnWDR/JGywvJB8IJMV/LB07ZNSEBEucAhDcf+AF95AhQunRGl157DejUySLCckUEREAEREAEfCQQPkGOPF7Wx2aSwlSk2GB+R3YCJE75H4xF4OdrFgwoCRALguC4EPgBffAgUKaMBIhFY0quiIAIiIAIxJBA+AQ5TtuBYti7/DXNJPM33gi1OWEC0KOHBEj+RiBfW5MAyVfc2TeWcALk9deB++6ziLBcEQEREAEREAEfCegiQv9gUsDxLhUmoPfrl5G0Hq2FsWOB3r39azuXlgI/X8tlf2NRXQIkFlTzaDPwAzpyBUQCJI8jQY+JgAiIgAgEgoAEiL9hevTR0JG8LNltwXr8ceC55/xtOxfWAj9fy0VfY1VVAiRWZPNgN/ADOlKATJ8e/dzuPLDRIyIgAiIgAiJgHQEJEH9D8vTTwIgROQuQKlUAXk4YpxL4+VqcuIU3KwFiQRAcFwI/oCMFyObNwBVXWERYroiACIiACIiAjwQkQHyECeC//wWuuy5nAaIVEH+5x8GaBEgcoGfVZOAFyIEDQNmyGd2L4wkVFoVVroiACIiACCQqAQkQfyO7bh1w1VVAuXIAb0+/4ALgxIlz29iwAahTx9+2c2Et8PO1XPQ1VlUlQGJFNg92Az+gI1dAJEDyMAr0iAiIgAiIQGAIXHllKHGaRf/neQ/b//4H1KwJlCwJ8Gh/3oB+6lRmu40aAZ9+6r0tDxYCP1/z0He/HpUA8YukD3YCP6B5gkWhQhkkdu8Gypf3gYxMiIAIiIAIiICFBNq0Ad5/XwLEr9Bs3w4wv4MXOvJix/POA3i/SmSJs9gL/HzNr3h5sCMB4gGe348GfkBHCpBZs4C77/Ybk+yJgAiIgAiIgB0EmAjdtStQqhQwb54dPgXZi6NHQ6dgceWjZ0+AN6JHig0ev8tjeONYAj9fiyM7p2kJEAuC4LgQ+AH9889A4cIZRCVALBpdckUEREAEREAEAkaAq0vLlwN/+lPI8aVLgSZNQsIkjiXw87U4spMAsQB+pAuBH9CHDwMpKRndevNN4K67LCQtl0RABERABERABAJBYNs24PLLQ67u3w+UKRN3twM/X4s7QUArIBYEwXEh8AOavxh4coVTZs8G7rzTIsJyRQREQAREQAREwGoCn30Wyv9gsjlzQVjatg19/Y9/WOF64OdrFlCUALEgCBIgFgVBroiACIiACIiACMSPQJEiwOnTwLffAh98ENpu1alTKCHdkiIB4j0QSS9Atm7diqFDh2Lx4sXYs2cPKlSogE6dOiE1NRVF+EMAXra5FVWrVj2H9qeffopGVOhny+zZszFw4EBTv0aNGhg1ahTaUrW7LIEf0JH3gLz1FvC737nsvaqJgAiIgAiIgAgkPYESJQAmo69eDVx9dQgH/33hhdagCfx8zQKSSS9AFi5ciFmzZqFjx46oXr061q1bh+7du6Nz58547rnnMgmQRYsWoW7duulhK1OmDM47q8iXL1+Opk2bYsSIEWjfvj1mzJhhBMiqVatQr149V6EO/IA+dAi46KKMvr79NnDHHa76rkoiIAIiIAIiIAIiYC405gvN//wntA2LhZcR8mQsS0rg52sWcEx6ARItBmPGjMHEiROxZcuWTALkiy++QIMGDaKGrUOHDjh27BgWLFiQ/jlXR1h/0qRJrkId+AEdKUB4oVD16q76rkoiIAIiIAIiIAIigMsuA3btAhYvBlq2DAH58UdtwUqwoSEBEiWgAwYMAFdGVq5cmUmAVKpUCSdPnkTNmjXx1FNP4dZbb01/unLlyujTpw9683zqs2XQoEGYO3cuVnMZ0UUJvAAJ34LFi4Qo4OJ8VJ4L7KoiAiIgAiIgAiJgC4Fq1UL5H3yh2759yCteRhh+0XGcfQ38fC3O/Ni8BEhEEDZv3oxrrrnGbL/iViyW/fv3Y9q0aWjSpAkKFiyIt99+G6NHjzbiwhEhzBeZOnWq2crllAkTJmDIkCHYu3dv1FCfOnUK/OMUDmiKnCNHjqBkyZIWDI9cupCWBlx8ceihl18GHnwwlwZUXQREQAREQAREIKkJ1KkDbNoE8CRN5yh/XnRcoIA1WCRAvIciYQVIv379TA5GdmXjxo2oXbt2epWdO3eiWbNmaN68OV7mBDqb0qVLF3z77bf4+OOPTa28CJDBgwcbgRJZAitAeGrF2cR9s4TKlR8Lzuv2/mMiCyIgAiIgAiIgAvlCgFvdOX+YOhW4//6Q8KAAsahIgHgPRsIKkLS0NBzglqBsSrVq1dJPutq1a5cRHszbmDJlilnpyK68+OKLGDZsGHbv3m2q5WULVsKtgIQLEEL5y1+AP/7R+yiVBREQAREQAREQgeQgwBfAZ3egpHf4zBmr+i4B4j0cCStAcoOGKx8tWrQwW69ef/11FHKxz5Dbsz7//HNzyhULk9CPHz+O+fPnpzfduHFj1K9fP3mS0Jkk5lwaRAozZgBhW9JyExPVFQEREAEREAERSEICr74KdOuWueMSIAk3EJJegFB8cOWjSpUqJocjXHyUL1/eBJzf5xarhg0bmn/PmTPH3PfBbVoPPPCA+R6P4eX2rZEjR6Jdu3aYOXMmhg8fnlzH8B48mHnL1fvvA61bJ9wPjTokAiIgAiIgAiIQIwLRcj0kQGIEO35mk16AcLuVIyIiw3Dm7ICnAGE+ybZt21C4cGGTN/Lkk0/izjvvzPQILyLkCVrORYRMVE+qiwj37QMuuSSDyaJFwE03xW90q2UREAEREAEREIFgEZAACVa88uht0guQPHKLyWOB31MYKUCWLAGaN48JKxkVAREQAREQARFIQAKVKwPffRfq2Lx5oXmEZSeDBn6+ZsGwkQCxIAiOC4Ef0OH3gLBTS5cCN95oEWG5IgIiIAIiIAIiYDWBF14AHnss5OJ11wHczl2qlFUuB36+ZgFNCRALgpAwAuTwYSAlJYPoJ58AjRtbRFiuiIAIiIAIiIAIWE2gc2fg9dczXGSu7SuvWOWyBIj3cEiAeGfom4XAD+hIAbJ1K8Ab0VVEQAREQAREQAREwA2BnTuBihUzaj77LPD0026ezLc6gZ+v5RuprBuSALEgCI4LgR/Q4adg/eY3wJtvWrdv06JwyxUREAEREAEREIFoBNq0CW29Yrn+eh41ahWnwM/XLKApAWJBEBJGgOzdC5w9uhhpaUDZshbRlSsiIAIiIAIiIAKBIHDrrYBzrxr/bt/eKrclQLyHQwLEO0PfLAR+QJ86BRQrFuJx5IhWP3wbGTIkAiIgAiIgAklE4K67gLfeCnX4q6+AmjWt6nzg52sW0JQAsSAICbMCIgFi0WiSKyIgAiIgAiIQUAKdOgHTp4ecP3QIKF3aqo5IgHgPhwSId4a+WQj8gD55Ejj//BAPrYD4Ni5kSAREQAREQASSisDddwOzZ4e6/NNPQKFCVnU/8PM1C2hKgFgQBMeFwA/o8HtA+PVFF1lEV66IgAiIgAiIgAgEgsBtt4UuIWThy82iRa1yO/DzNQtoSoBYEISEESDhSej79wNlylhEV66IgAiIgAiIgAgEgkD4CsiZM9a5LAHiPSQSIN4Z+mYh8ANaAsS3sSBDIiACIiACIpC0BFauBH7969B9IN99Zx2GwM/XLCAqAWJBEBJmBYSrHuXKhbrz/fdAiRIW0ZUrIiACIiACIiACgSCwdCnQrBlQqxawaZN1LkuAeA+JBIh3hr5ZCPyAZuK5c1IFT8QqUsQ3NjIkAiIgAiIgAiKQJAQWLgRuvhlo2BBYtcq6Tgd+vmYBUQkQC4KQMCsgEiAWjSa5IgIiIAIiIAIBJTB+PNCrV8h55YAENIjZuy0BYlFYA6+oeVa3c/LViRMZlxJaxFiuiIAIiIAIiIAIWE6ALzR79ADuvde6W9BJLvDzNQvCLwFiQRASZgVk926gQoVQdw4eBFJSLKIrV0RABERABERABETAOwEJEO8MJUC8M/TNQuAH9LFjQPHiIR4//JDxtW+EZEgEREAEREAEREAE4ksg8PO1+OIzrUuAWBAEx4XAD+jjx4ELLwx15+jRjK8tYixXREAEREAEREAERMALgcDP17x03qdnJUB8AumHmcAP6PAVEAkQP4aEbIiACIiACIiACFhGIPDzNQt4SoBYEISEWQEJvwfk8GGgVCmL6MoVERABERABERABEfBOQALEO0MJEO8MfbMQ+AGtJHTfxoIMiYAIiIAIiIAI2Ekg8PM1C7BKgFgQhIRZAZEAsWg0yRUREAEREAEREIFYEJAA8U5VAsQ7Q98sBH5Ap6UBF18c4vH990CJEr6xkSEREAEREAEREAERsIFA4OdrFkCUALEgCAmzAkLR4eR9nDwJFC1qEV25IgIiIAIiIAIiIALeCUiAeGcoAeKdoW8WAj+gJUB8GwsyJAIiIAIiIAIiYCeBwM/XLMAqAQLg1ltvxZdffol9+/YhJSUFrVq1wqhRo1DBudUbwJo1a9CzZ0+sWLEC5cqVQ69evfDUU09lCuHs2bMxcOBAbN26FTVq1DA22rZt6zrMgR/QPPnKuf2cd4Kcf77rvquiCIiACIiACIiACASBQODnaxZAlgABMHbsWFx//fW49NJLsXPnTjzxxBMmNMuXLzd/c6DVrFnTCJP+/ftj7dq16NatG8aNG4eHH344vW7Tpk0xYsQItG/fHjNmzDACZNWqVahXr56rUAd+QO/cCVSsGOrroUNA6dKu+q1KIiACIiACIiACIhAUAoGfr1kAWgIkShDeeecd3HbbbTh16hTOO+88TJw4EampqdizZw+KFClinujXrx/mzp2LTZs2mX936NABx44dw4IFC9ItNmrUCA0aNMCkSZNchTrwA/rIkQzRwUsJL7jAVb9VSQREQAREQAREQASCQiDw8zULQEuARATh4MGD6NGjh1kJWbZsmfm0S5cuZhWEgsMpS5YsQcuWLcH63LZVuXJl9OnTB717906vM2jQIPPM6tWrXYU68AP6hx+AkiVDfdUWLFcxVyUREAEREAEREIFgEQj8fM0C3BIgZ4PQt29fjB8/HsePHwdXLriSUaZMGfNp69atUbVqVUyePDk9ZBs2bEDdunXBv+vUqWNWRqZOnYqOHTum15kwYQKGDBmCvXv3Rg01V1j4xykc0JUqVcKRI0dQ0pnIWzBIXLsQLkBOnACKFXP9qCqKgAiIgAiIgAiIQBAISIB4j1LCChBukWIORnZl48aNqF27tqmyf/9+s5qxbds2IxpKlSplREiBAgViJkAGDx5s2oosgRUgugfE+0+kLIiACIiACIiACFhNQALEe3gSVoCkpaXhwIED2RKqVq1aek5HeMUdO3aYlQgmoTM5PVZbsBJuBWTXLuCyy0IolYTu/adTFkRABERABERABKwjIAHiPSQJK0C8oNm+fTuqVKkC5nk0b948PQmdW6mYlM7y9NNPY86cOZmS0Ll9a/78+elNN27cGPXr10+eJPQzZ4AmTYDChYGPPgIKFPASBj0rAiIgAiIgAiIgAtYRkADxHpKkFyCfffaZudvjhhtuMMnk33zzjbnLg2Jj/fr1KFq0qMnJqFWrltmKxVyRdevWmWN4eXxv+DG8zZo1w8iRI9GuXTvMnDkTw4cPT65jeDkeKUJYJD68/3TKggiIgAiIgAiIgHUEJEC8hyTpBQjv9HjsscfMSVU8Rpd3gbRp0wYDBgzAZc52ooiLCMuWLWsuIqQYCS+8iJDPORcRjh49OrkuIvQ+HmVBBERABERABERABKwmIAHiPTxJL0C8I/TPgga0fyxlSQREQAREQAREQARiQUDzNe9UJUC8M/TNgga0byhlSAREQAREQAREQARiQkDzNe9YJUC8M/TNgga0byhlSAREQAREQAREQARiQkDzNe9YJUC8M/TNgga0byhlSAREQAREQAREQARiQkDzNe9YJUC8M/TNgga0byhlSAREQAREQAREQARiQkDzNe9YJUC8M/TNgga0byhlSAREQAREQAREQARiQkDzNe9YJUC8M/TNgga0byhlSAREQAREQAREQARiQkDzNe9YJUC8M/TNgga0byhlSAREQAREQAREQARiQkDzNe9YJUC8M/TNgga0byhlSAREQAREQAREQARiQkDzNe9YJUC8M/TNgga0byhlSAREQAREQAREQARiQkDzNe9YJUC8M/TNwpEjR1C6dGl89913KFmypG92ZUgEREAEREAEREAERMAfAhQg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97Njxc8ZL96BElkogpzTtPz7kZMlERABERABERABEUhuAkkhQLyEmNuuuDrCN+7hKwlebPJZHnlLezz9KfxWdq929bwIiIAIiIAIiIAIiIAI2ExAAiSH6Hz99dd4//33zUlZfhae5MT8ktTU1EwJ2362IVsiIAIiIAIiIAIiIAIiYBsBCRDbIiJ/REAEREAEREAEREAERCCBCUiAJHBw1TUREAEREAEREAEREAERsI2ABIhtEZE/IiACIiACIiACIiACIpDABCRAEji46poIiIAIiIAIiIAIiIAI2EZAAsS2iMgfERABERABERABERABEUhgAhIgCRxcdU0EREAEREAEREAEREAEbCMgAWJbROSPCIiACIiACIiACIiACCQwAQmQBA6uuiYCIiACfhHo2rUrpk6daswVLlwYF110EerXr4+OHTuCnxUsWNBVU1OmTEHv3r1x+PBhV/VVSQREQAREIPEISIAkXkzVIxEQARHwnQBFxt69e/Hqq6/i559/Nl8vXLgQI0aMwI033oh33nnHCJOcigRIToT0uQiIgAgkPgEJkMSPsXooAiIgAp4JUIBw1WLu3LmZbC1evBg33XQTXnrpJTz00EN4/vnnjUjZsmWLWSW55ZZbMHr0aBQvXhwffvghWrRoken5QYMGYfDgwTh16hRSU1PxxhtvmHbq1auHUaNGoXnz5p59lwEREAEREAG7CEiA2BUPeSMCIiACVhLImmWY4QAAAvNJREFUSoDQ2QYNGqBChQp47733MG7cOFx99dWoWrWqESG///3v0bJlS0yYMAE//vgjJk6ciGeeeQZfffWV6SeFCf90794dGzZswMiRI42tf/zjHxgwYADWrl2LGjVqWMlETomACIiACOSNgARI3rjpKREQARFIKgLZCZB77rkHa9asMQIisrz11lt49NFHsX//fvNRtC1Y27dvR7Vq1cC/KT6c0qpVK1x77bUYPnx4UrFWZ0VABEQg0QlIgCR6hNU/ERABEfCBQHYCpEOHDli3bh3Wr1+PRYsWmbyQTZs24fvvv8dPP/2EkydP4tixY7jggguiCpB3330X7du3x4UXXpjJU27LuuOOOzBr1iwfeiATIiACIiACthCQALElEvJDBERABCwmkJ0A4WlYlStXxvjx41G7dm306NEDFCXMAVm2bBkefPBBHDp0CKVLl44qQCgw7rvvPiNgChUqlIkCt2eVL1/eYjJyTQREQAREILcEJEByS0z1RUAERCAJCeSUhP7KK6+gZMmS5lherng4x/IOGzYMAwcOTBcgM2bMwCOPPIIffvghneLXX3+NWrVqYenSpeZELRUREAEREIHEJiABktjxVe9EQAREwBcC2R3Dy5OqeDoWt2ExIZ2J6Dz96pNPPkH//v2xc+fOdAGyfPlyNGnSxGzVYrI6t2XxT6dOnUz9P//5z2jYsCHS0tLwwQcfmLtG2rVr50sfZEQEREAERMAOAhIgdsRBXoiACIiA1QQiLyJMSUkxAuLee+/F/fffn77iMXbsWIwZM8Ycpdu0aVOztapLly7pAoSd5Bat2bNn48CBA3CO4T19+jS4WjJt2jQjWMqWLYtGjRphyJAhuOqqq6xmI+dEQAREQARyR0ACJHe8VFsEREAEREAEREAEREAERMADAQkQD/D0qAiIgAiIgAiIgAiIgAiIQO4ISIDkjpdqi4AIiIAIiIAIiIAIiIAIeCAgAeIBnh4VAREQAREQAREQAREQARHIHYH/B8cMXU1gVinmAAAAAElFTkSuQmCC" width="800">


    c:\users\crazy\appdata\local\programs\python\python38-32\lib\site-packages\pandas\plotting\_matplotlib\core.py:1235: UserWarning: FixedFormatter should only be used together with FixedLocator
      ax.set_xticklabels(xticklabels)
    




    <AxesSubplot:xlabel='Date'>




```python
idx = silver_rate.index
```


```python
idx
```




    Index(['08/20/2020 14:53:01', '08/20/2020 14:54:01', '08/20/2020 14:55:01',
           '08/20/2020 14:56:01', '08/20/2020 14:57:01', '08/20/2020 14:58:01',
           '08/20/2020 14:59:01', '08/20/2020 15:00:01', '08/20/2020 15:01:01',
           '08/20/2020 15:02:01',
           ...
           '09/30/2020 15:20:45', '09/30/2020 15:21:45', '09/30/2020 15:22:45',
           '09/30/2020 15:23:45', '09/30/2020 15:24:45', '09/30/2020 15:25:45',
           '09/30/2020 15:26:45', '09/30/2020 15:27:45', '09/30/2020 15:28:45',
           '09/30/2020 15:29:45'],
          dtype='object', name='Date', length=22077)




```python
idx = silver_rate.loc['08/20/2020 14:53:01':'08/20/2020 15:02:01'].index
```


```python
rates = silver_rate.loc['08/20/2020 14:53:01':'08/20/2020 15:02:01']['dBid_close']
```


```python
rates.plot()
```




    <AxesSubplot:xlabel='Date'>




```python
fig, ax = plt.subplots()
ax.plot_date(idx, rates, '-')

ax.xaxis.grid(True)
ax.yaxis.grid(True)

fig.autofmt_xdate()
plt.tight_layout()
```


    <IPython.core.display.Javascript object>



<img src="data:image/png;base64,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" width="640">



```python

```
