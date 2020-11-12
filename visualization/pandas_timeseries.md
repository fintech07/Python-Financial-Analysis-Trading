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



<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAABLAAAAJYCAYAAABy5h8aAAAgAElEQVR4XuzdCbgsRXk/4AIXXBDcERQMKrhdFfdd0RgXMK5xSxQElUgQRWNYVASishpFRQW3QGKIiHFD4xbFLSSKogKCJsYAIqJEAhrIHxf4P9U359zL5eKt7ts1U1/Pe57HhwRqer55v5rqrt+Z6bPBVVdddVXyQ4AAAQIECBAgQIAAAQIECBAgQKBRgQ0EWI12RlkECBAgQIAAAQIECBAgQIAAAQKdgADLRCBAgAABAgQIECBAgAABAgQIEGhaQIDVdHsUR4AAAQIECBAgQIAAAQIECBAgIMAyBwgQIECAAAECBAgQIECAAAECBJoWEGA13R7FESBAgAABAgQIECBAgAABAgQICLDMAQIECBAgQIAAAQIECBAgQIAAgaYFBFhNt0dxBAgQIECAAAECBAgQIECAAAECAixzgAABAgQIECBAgAABAgQIECBAoGkBAVbT7VEcAQIECBAgQIAAAQIECBAgQICAAMscIECAAAECBAgQIECAAAECBAgQaFpAgNV0exRHgAABAgQIECBAgAABAgQIECAgwDIHCBAgQIAAAQIECBAgQIAAAQIEmhYQYDXdHsURIECAAAECBAgQIECAAAECBAgIsMwBAgQIECBAgAABAgQIECBAgACBpgUEWE23R3EECBAgQIAAAQIECBAgQIAAAQICLHOAAAECBAgQIECAAAECBAgQIECgaQEBVtPtURwBAgQIECBAgAABAgQIECBAgIAAyxwgQIAAAQIECBAgQIAAAQIECBBoWkCA1XR7FEeAAAECBAgQIECAAAECBAgQICDAMgcIECBAgAABAgQIECBAgAABAgSaFhBgNd0exREgQIAAAQIECBAgQIAAAQIECAiwzAECBAgQIECAAAECBAgQIECAAIGmBQRYTbdHcQQIECBAgAABAgQIECBAgAABAgIsc4AAAQIECBAgQIAAAQIECBAgQKBpAQFW0+1RHAECBAgQIECAAAECBAgQIECAgADLHCBAgAABAgQIECBAgAABAgQIEGhaQIDVdHsUR4AAAQIECBAgQIAAAQIECBAgIMAyBwgQIECAAAECBAgQIECAAAECBJoWEGA13R7FESBAgAABAgQIECBAgAABAgQICLDMAQIECBAgQIAAAQIECBAgQIAAgaYFBFhNt0dxBAgQIECAAAECBAgQIECAAAECAixzgAABAgQIECBAgAABAgQIECBAoGkBAVbT7VEcAQIECBAgQIAAAQIECBAgQICAAMscIECAAAECBAgQIECAAAECBAgQaFpAgNV0exRHgAABAgQIECBAgAABAgQIECAgwDIHCBAgQIAAAQIECBAgQIAAAQIEmhYQYDXdHsURIECAAAECBAgQIECAAAECBAgIsMwBAgQIECBAgAABAgQIECBAgACBpgUEWE23R3EECBAgQIAAAQIECBAgQIAAAQICLHOAAAECBAgQIECAAAECBAgQIECgaQEBVtPtURwBAgQIECBAgAABAgQIECBAgIAAyxwgQIAAAQIECBAgQIAAAQIECBBoWkCA1XR7FEeAAAECBAgQIECAAAECBAgQICDAMgcIECBAgAABAgQIECBAgAABAgSaFhBgNd0exREgQIAAAQIECBAgQIAAAQIECAiwzAECBAgQIECAAAECBAgQIECAAIGmBQRYTbdHcQQIECBAgAABAgQIECBAgAABAgIsc4AAAQIECBAgQIAAAQIECBAgQKBpAQFW0+1RHAECBAgQIECAAAECBAgQIECAgADLHCBAgAABAgQIECBAgAABAgQIEGhaQIDVdHsUR4AAAQIECBAgQIAAAQIECBAgIMAyBwgQIECAAAECBAgQIECAAAECBJoWEGA13R7FESBAgAABAgQIECBAgAABAgQICLDMAQIECBAgQIAAAQIECBAgQIAAgaYFBFhNt0dxBAgQIECAAAECBAgQIECAAAECAixzgAABAgQIECBAgAABAgQIECBAoGkBAVbT7VEcAQIECBAgQIAAAQIECBAgQICAAMscIECAAAECBAgQIECAAAECBAgQaFpAgNV0exRHgAABAgQIECBAgAABAgQIECAgwDIHCBAgQIAAAQIECBAgQIAAAQIEmhYQYDXdHsURIECAAAECBAgQIECAAAECBAgIsMwBAgQIECBAgAABAgQIECBAgACBpgUEWE23R3EECBAgQIAAAQIECBAgQIAAAQICLHOAAAECBAgQIECAAAECBAgQIECgaQEBVtPtURwBAgQIECBAgAABAgQIECBAgIAAyxwgQIAAAQIECBAgQIAAAQIECBBoWkCA1XR7FEeAAAECBAgQIECAAAECBAgQICDAMgcIECBAgAABAgQIECBAgAABAgSaFhBgNd0exREgQIAAAQIECBAgQIAAAQIECAiwzAECBAgQIECAAAECBAgQIECAAIGmBQRYTbenX3Ff/vKX0xFHHJG++c1vpp/85CfpIx/5SHrKU56yfJCrrroqHXDAAend7353uuSSS9JDH/rQ9M53vjNts802y2MuvvjitOeee6aTTjopbbjhhunpT396estb3pI23njj4mKuvPLKdMEFF6Sb3OQmaYMNNih+nIEECBAgQIAAAQIECBAgQGCIQN7v/vKXv0xbbLFFt5f1Mz0BAdaEevqpT30q/fM//3O6733vm572tKddI8A67LDD0iGHHJKOO+64tPXWW6f9998/nXHGGemss85KN7jBDTqJJzzhCV34dcwxx6Rf//rXaZdddkn3v//90/HHH18sdf7556ctt9yyeLyBBAgQIECAAAECBAgQIEBgDIEf/ehH6Xa3u90Yh3KMxgQEWI01ZKxy8iefVv8EVk6jcxL953/+5+mVr3xl9zSXXnpp2myzzdKxxx6bnv3sZ6ezzz473e1ud0unnnpqut/97teN+fSnP5122GGHlEOp/PiSn3zcm970pikvHJtssknJQ4whQIAAAQIECBAgQIAAAQKDBX7xi190H6TI3zbadNNNBx/HA9sVEGC125v1qmzNAOuHP/xhuuMd75i+9a1vpe2222752I985CO7/z9/TfB973tfF3D993//9/J//81vftN9OuvEE09MT33qU4tqygtHXjBykCXAKiIziAABAgQIECBAgAABAgTWQ8A+dD3wgjxUgBWkUX3LXDPAOuWUU7p7XuV7U22++ebLh3vmM5/Z3afqhBNOSAcffHD39cLvf//7V3u6W9/61umggw5Ku++++1rLuOKKK1L+39LPUvItwOrbNeMJECBAgAABAgQIECBAYIiAAGuIWqzHCLBi9au42lkGWAceeGAXcK35I8AqbpeBBAgQIECAAAECBAgQILAeAgKs9cAL8lABVpBG9S1zll8h9Amsvt0xngABAgQIECBAgAABAgTGFBBgjanZ5rEEWG32Zb2rurabuOcbuOf7XOWf/AbPXw9c8ybu3/jGN7q/ZJh/PvvZz6bHP/7xvW7ibuFY7/Y5AAECBAgQIECAAAECBAj0ELAP7YEVdKgAK2jj1lb2//zP/6Qf/OAH3X+6973vnd70pjelRz3qUenmN7952mqrrdJhhx2WDj300O4+V1tvvXXaf//90+mnn57OOuus7kbt+ecJT3hC+ulPf5qOPvro9Otf/zrtsssu3V8kPP7444ulLBzFVAYSIECAAAECBAgQIECAwAgC9qEjIDZ+CAFW4w3qU94Xv/jFLrBa82fnnXfuPmV11VVXpQMOOCC9613v6v606MMe9rD0jne8I2277bbLD7n44ovTS17yknTSSSelDTfcMD396U9Pb33rW9PGG29cXIqFo5jKQAIECBAgQIAAAQIECBAYQcA+dATExg8hwGq8QRHLs3BE7JqaCRAgQIAAAQIECBAgEFfAPjRu70orF2CVShlXLGDhKKYykAABAgQIECBAgAABAgRGELAPHQGx8UMIsBpvUMTyLBwRu6ZmAgQIECBAgAABAgQIxBWwD43bu9LKBVilUsYVC1g4iqkMJECAAAECBAgQIECAAIERBOxDR0Bs/BACrMYbFLE8C0fErqmZAAECBAgQIECAAAECcQXsQ+P2rrRyAVaplHHFAhaOYioDCRAgQIAAAQIECBAgQGAEAfvQERAbP4QAq/EGRSzPwhGxa2omQIAAAQIECBAgQIBAXAH70Li9K61cgFUqZVyxgIWjmMpAAgQIECBAgAABAgQIEBhBwD50BMTGDyHAarxBEcuzcETsmpoJECBAgAABAgQIECAQV8A+NG7vSisXYJVKGVcsYOEopjKQAAECBAgQIECAAAECBEYQsA8dAbHxQwiwGm9QxPIsHBG7pmYCBAgQIECAAAECBAjEFbAPjdu70soFWKVSxhULWDiKqQwkQIAAAQIECBAgQIAAgREE7ENHQGz8EAKsxhsUsTwLR8SuqZkAAQIECBAgQIAAAQJxBexD4/autHIBVqmUccUCFo5iKgMJECBAgAABAgQIECBAYAQB+9AREBs/hACr8QZFLM/CEbFraiZAgAABAgQIECBAgEBcAfvQuL0rrVyAVSplXLGAhaOYykACBAgQIECAAAECBAgQGEHAPnQExMYPIcBqvEERy7NwROyamgkQIECAAAECBAgQIBBXwD40bu9KKxdglUoZVyxg4SimMpAAAQIECBAgQIAAAQIERhCwDx0BsfFDCLAab1DE8iwcEbumZgIECBAgQIAAAQIECMQVsA+N27vSygVYpVLGFQtYOIqpDCRAgAABAgQIECBAgACBEQTsQ0dAbPwQAqzG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0ylk6gtHrYVZgDWd94BXQoAAAQIECBDoK1DrGlOAJcBacy5OdU5MfR/ad02Z4ngB1hS7OufXNPWFo9bFhQBrzhPX0xMgQIAAAQIE5ihQ6xpzqmFF31ZF882vL1rN86536vvQvnN+iuMFWFPs6pxf09QXjloLswBrzhPX0xMgQIAAAQIE5ihQ6xpTgLWyqdF8I9Y8b+Op70PnuDw189QCrGZaMZ1Cpr5w1FqYBVjTeQ94JQQIECBAgACBvgK1rjEFWAKsNefiVOfE1PehfdeUKY4XYE2xq3N+TVNfOGpdXAiw5jxxPT0BAgQIECBAYI4Cta4xpxpW9G1VNN/8+qLVPO96p74P7TvnpzhegDXFrs75NU194ai1MAuw5jxxPT0BAgQIECBAYI4Cta4xBVgrmxrNN2LN8zae+j50jstTM08twGqmFdMpZOoLR62FWYA1nfeAV0KAAAECBAgQmLpArWtigduqmcN4pUXpnJj6PnTqa0rJ6xNglSgZ00tg6gtHrRNJn8W5V0OC/sap72s0ngABAgQIECBAYHYCta6JS8OKvq80Wr359UWred71Tn0f2nfOT3G8AGuKXZ3za5r6wlFrYRZgzXnienoCBAgQIECAAIFigVrXxAKsVS1gvNKidE5MfR9a/Oac8EAB1oSbO6+XNvWFo9aJpM/i3Le3tWouPZn0rdd4AgQIECBAgACBtgWiXV9Gqzd3P1rN86536vvQtleE2VQnwJqN80ye5Q1veEP65Cc/mb797W+n61//+umSSy65xvOed955affdd08nn3xy2njjjdPOO++cDjnkkHTd6153eewXv/jF9IpXvCJ997vfTVtuuWV6zWtek57//OcXv4apLxy1FmYBVvEUM5AAAQIECBAgQGDOArWuiWv9gjRavQKsVRO8dE5MfR8657d8E08vwGqiDeMUccABB6Sb3vSm6fzzz0/vfe97rxFg/fa3v03bbbddus1tbpOOOOKI9JOf/CTttNNO6UUvelE6+OCDuyL+8z//M61YsSK9+MUvTi984QvT5z//+bTXXnt1wdjjHve4okKnvnDUOvkJsIqml0EECBAgQIAAAQINCNS6Ji4NK/oSRKtXgCXA6jvHF2G8AGuCXT722GO70GnNT2B96lOfSk984hPTBRdckDbbbLPulR999NFpn332SRdddFH3qa38f+ew6swzz1yWefazn90d69Of/nSRlgCriGmtg5ywh9t5JAECBAgQIECAwOwEogVC0eoVYAmwZvdujvNMAqw4vSqu9NoCrNe+9rXp4x//ePcVw6Wf/ImrO9zhDum0005L9773vdMjHvGIdJ/73CcdeeSRy2P++q//ugvELr300rXWcMUVV6T8v6WfHGDlrx7m8Ztssklx3VEG1jr55dcvwIoyC9RJgAABAgQIEFhsgVrXxK6HV80rxistSufE1D9IsdgrzspXL8Ca4Cy4tgBrt912S+eee276zGc+s/yqL7/88nTjG984/eM//mN6whOekLbddtu0yy67pP322295TP5vO+64Y8pjb3jDG15D7MADD0wHHXTQNf69AKv/5CpdnPseOdrJr+/rM54AAQIECBAgQGC2AtGuL6PVm7sZreZ51yvAmu0aMI9nE2DNQ73Hc+67777psMMO+52POPvss9Nd7nKX5TGzDrB8AqtHQ9cxVIA1nqUjESBAgAABAgQI1BOYd1jR95VFq1eAtarDpXskAVbfd0W88QKsxnuW703185///HdWmb8CmO9ftfQz668Qrlnc1BeOWie/7Fi6OPedtrVqrlVv39dnPAECBAgQIECAwGwFol1fRqtXgCXAmu07OsazCbBi9KlXleu6iXv+64O3vvWtu2O+613vSn/xF3+Rfvazn6WNNtqou4l7/srgGWecsfycf/zHf5wuvvhiN3H/P5FaJz8BVq9pbjABAgQIECBAgMAcBWpdE9f6BWm0egVYAqw5vr2bfWoBVrOt6V/Yeeed1wVN+UbtRxxxRPrKV77SHeROd7pT2njjjdNvf/vbtN1226UtttgiHX744enCCy9Mz3ve89ILX/jCdPDBB3dj803dV6xYkfbYY4+06667pi984QvppS99afeXCR/3uMcVFeUTWEVMax3khD3cziMJECBAgAABAgRmJxAtEIpWrwBLgDW7d3OcZxJgxenVOit9/vOfn4477rhrjDv55JPT9ttv3/37fBP33XffPX3xi1/sbt6+8847p0MPPTRd97rXXX5c/m8vf/nL01lnnZVud7vbpf333z/lY5f+CLBKpa45ToA13M4jCRAgQIAAAQIEZicQLRCKVq8AS4A1u3dznGcSYMXpVZhKBVjDWyXAGm7nkQQIECBAgAABArMTiBYIRatXgCXAmt27Oc4zCbDi9CpMpQKs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AQIECBAgQKC2gACrtrDjC7DMgdEFBFijkzogAQIECBAgQIAAAQIEmhYQYDXdnkkUJ8CaRBvbehECrLb6oRoCBAgQIECAAAECBAjUFhBg1RZ2fAGWOTC6gABrdFIHJECAAAECBAgQIECAQNMCAqym2zOJ4gRYk2hjWy9CgNVWP1RDgAABAgQIECBAgACB2gICrNrCji/AMgdGFxBgjU7qgAQIECBAgAABAgQIEGhaQIDVdHsmUZwAaxJtbOtFCLDa6odqCBAgQIAAAQIECBAgUFtAgFVb2PEFWObA6AICrNFJHZAAAQIECBAgQIAAAQJNCwiwmm7PJIoTYE2ijW29CAFWW/1QDQECBAgQIECAAAECBGoLCLBqCzu+AMscGF1AgDU6qQMSIECAAAECBAgQIECgaQEBVtPtmURxAqxJtLGtFyHAaqsfqiFAgAABAgQIECBAgEBtAQFWbWHHF2CZA6MLCLBGJ3VAAgQIECBAgAABAgQINC0gwGq6PZMoToA1iTamdM4556TXve516Qtf+EK68MIL0xZbbJGe+9znple/+tXp+te//vKrPP3009Mee+yRTj311HSrW90q7bnnnmnvvfe+msKJJ56Y9t9//+6Y22yzTTrssMPSDjvsUCwlwCqmMpAAAQIECBAgQIAAAQKTEBBgTaKNTb8IAVbT7Skv7tOf/nQ64YQT0nOe85x0pzvdKZ155pnpRS96UXre856X3vjGN3YHysHStttumx7zmMek/fbbL51xxhlp1113TUceeWTabbfdujGnnHJKesQjHpEOOeSQ9MQnPjEdf/zxXYB12mmnpRUrVhQVJMAqYjKIAAECBAgQIECAAAECkxEQYE2mlc2+EAFWs61Z/8KOOOKI9M53vjP98Ic/7A6W/+/8iaz8Ca2lT2Xtu+++6aMf/Wj63ve+14151rOelS677LL0iU98YrmABz3oQWm77bZLRx99dFFRAqwiJoMIECBAgAABAgQIECAwGQEB1mRa2ewLEWA125r1L+w1r3lNyp/M+sY3vtEdbKedduo+hZUDq6Wfk08+OT360Y9OF198cbrZzW6Wttpqq/SKV7wi7bXXXstjDjjggO4x3/nOd4qKEmAVMRlEgAABAgQIECBAgACByQgIsCbTymZfiACr2dasX2E/+MEP0n3ve9/u64P5q4T557GPfWzaeuut0zHHHLN88LPOOivd/e53T/mfd73rXbtPZh133HHdVxGXft7xjnekgw46KP30pz9da1FXXHFFyv9b+skB1pZbbpkuvfTStMkmm6zfC/FoAgQIECBAgAABAgQIEGheQIDVfIvCFyjAaryF+St++R5Uv+vn7LPPTne5y12Wh/z4xz9Oj3zkI9P222+f3vOe9yz/+1oB1oEHHtgFXGv+CLAan1zKI0CAAAECBAgQIECAwEgCAqyRIB3mWgUEWI1Pjosuuij9/Oc//51V3uEOd1i+p9UFF1zQBVf5vlXHHnts2nDDDZcfW+srhD6B1fgkUh4BAgQIxXj1bgAAIABJREFUECBAgAABAgQqCwiwKgM7fBJgTWgS5E9ePepRj+q+Ovj+978/Xec617naq1u6iXv+KuD1rne97r+96lWvSh/+8IevdhP3yy+/PJ100knLj33IQx6S7nnPe7qJ+4TmipdCgAABAgQIECBAgACBMQUEWGNqOtbaBARYE5kXObzKn7y6/e1v393DavXw6ja3uU33KvNX+u585zt398LaZ5990plnnpl23XXX9OY3vznttttu3ZhTTjml+/rhoYcemnbcccf0gQ98IB188MHptNNOSytWrCjSchP3IiaDCBAgQIAAAQIECBAgMBkBAdZkWtnsCxFgNduafoXlrwvusssua33QVVddtfzvTz/99LTHHnukU089Nd3ylrdMe+65Zxdmrf5z4oknpvwXDM8555y0zTbbpMMPPzztsMMOxQUJsIqpDCRAgAABAgQIECBAgMAkBARYk2hj0y9CgNV0e2IWJ8CK2TdVEyBAgAABAgQIECBAYKiAAGuonMeVCgiwSqWMKxYQYBVTGUiAAAECBAgQIECAAAECIwjYh46A2PghBFiNNyhieRaOiF1TMwECBAgQIECAAAECBOIK2IfG7V1p5QKsUinjigUsHMVUBhIgQIAAAQIECBAgQIDACAL2oSMgNn4IAVbjDYpYnoUjYtfUTIAAAQIECBAgQIAAgbgC9qFxe1dauQCrVMq4YgELRzGVgQQIECBAgAABAgQIECAwgoB96AiIjR9CgNV4gyKWZ+GI2DU1EyBAgAABAgQIECBAIK6AfWjc3pVWLsAqlTKuWMDCUUxlIAECBAgQIECAAAECBAiMIGAfOgJi44cQYDXeoIjlWTgidk3NBAgQIECAAAECBAgQiCtgHxq3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179+d/h99903ffSjH03f+973uv//Wc96VrrsssvSJz7xieWnf9CDHpS22267dPTRRxeVZOEoYjKIAAECBAgQIECAAAECBEYSsA8dCbLhwwiwGm7O+pR28cUXp9133737JNZXv/rV7lA77bRT9ymsHFgt/Zx88snp0Y9+dMrj89cOt9pqq/SKV7wi7bXXXstjDjjggO4x3/nOd9ZaUg7I8v+WfvJzbLnllunSSy9Nm2yyyfq8DI8lQIAAAQIECBAgQIAAAQLrFBBgrZMo/AABVvgWXv0F7LPPPumoo45Kl19+ecqfnMqfpLrFLW7RDXrsYx+btt5663TMMccsP+iss85Kd7/73VP+513vetfuk1nHHXdces5znrM85h3veEc66KCD0k9/+tO1ah144IHdf1/zR4A1scnl5RAgQIAAAQIECBAgQKBRAQFWo40ZsSwB1oiYNQ6Vv+KX70H1u37OPvvsdJe73KUb8l//9V/dp6nOPffcLlTadNNNuxBrgw02qBZg+QRWjc47JgECBAgQIECAAAECBAiUCgiwSqXijhNgNd67iy66KP385z//nVXe4Q53WL6n1eoDzz///O6rfPkm7vnm7rW+QrhmcRaOxieV8ggQIECAAAECBAgQIDAxAfvQiTV0LS9HgDXhHp933nnp9re/fcr3udp+++2Xb+KevwqYb+qef171qlelD3/4w1e7iXv++uFJJ520LPOQhzwk3fOe93QT9wnPFS+NAAECBAgQIECAAAECkQUEWJG7V1a7AKvMqflRX/va19Kpp56aHvawh3U3Y/+P//iPtP/++3f3rfrud7+bNtpoo+6m6ne+8527rxLme2WdeeaZadddd01vfvOb02677da9xvxprUc+8pHp0EMPTTvuuGP6wAc+kA4++OB02mmnpRUrVhQ5WDiKmAwiQIAAAQIECBAgQIAAgZEE7ENHgmz4MAKshpvTp7QzzjgjvexlL+v+UuBll12WNt988/T4xz8+veY1r0m3ve1tlw91+umnpz322KMLu255y1umPffcswuzVv858cQTu8edc845aZtttkmHH3542mGHHYrLsXAUUxlIgAABAgQIECBAgAABAiMI2IeOgNj4IQRYjTcoYnkWjohdUzMBAgQIECBAgAABAgTiCtiHxu1daeUCrFIp44oFLBzFVAYSIECAAAECBAgQIECAwAgC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Ln2y6u1vf3s65JBD0vnnn79cWP4NyfOe97wuuPjKV77SRMHR6s1oEWvOdX/oQx9KO++8c7rsssuWe5/vuZLnxOmnn56+/vWvpxvc4AZNzIto9ZoTdadNNN9o9VrX6s7f1Y+ew6rHPvax3Tp8/etfP/32t7/tPnX1yle+Mp1wwgnpC1/4Qtpmm21mV1DBMzl3FCCtx5BovubwejR7HQ+Ndu6IVq9zXb2568jTERBgTaeXTb+Sd73rXd2NYO9///svX/jmf5e/NviWt7wlPexhD1uuP39VYccdd0yf/OQn02Me85i5fAorWr0ZL1rN+dN2+cbA97nPfdKDH/zgrv+f+9znuq8O5mDzaU972nLvf/CDH6R73OMe6X3ve196znOeM5evl0ar15yovyRGe89Fq9ccrj+H81qbN3grVqxIT37yk7sn/Pd///fu3Lvvvvt298H6zW9+033iKv9zs802677iv88++8xlHc71RVuL1Vt3HpvDdX0jrsPOdfXnRLR1rb6IZ5ilgABrltoL+Fz564H50zP5K4L5ng/5t7n5gvjlL39594maZz/72en5z39+2muvvbr/ln9++tOfphe96EXpZje7WTruuONmqhat3owTreYvfelLXd+32GKLbkN04YUXpl122aX76uh5553X/d93vvOd0xFHHNHdHy1vrvL/8pz4z//8z5S/ejHLn2j1mhP1Z0e091y0es3h+nP4a1/7WnrGM57R/fGUTTfdtPvafv7a4Jve9KbuFwsvfvGL08UXX5z+/u//Pt30pjft/hhIPkfne2LlT+P827/9W/0i13iGaGuxeutOEXO4rm/Eddi5rv6ciLau1RfxDPMQEGDNQ32BnvOP/uiPur9gdPTRR3cXvB//+Me7397mf+abdecwK184H3744cv32sg8+XE3uclN0l//9V/PVCtavUtWkYx33XXX7iuif/d3f9eFlV/+8pe7vzL41re+tfv01WGHHdZtkF72spel5z73ucv9//M///N05plndvf1yp/mm9VPtHrNifozI9o6Ea1ec7j+HM6/RMp/ICOfi//f//t/6Vvf+lZ3Ts73t3r961/f3Ycwn5fz3MmfxMpf5c5/XCOv0+9973vTpz/96bT55pvXL3S1Z4i2Fqu37vQwh+v6RlyHnevqz4lo61p9Ec8wDwEB1jzUF+Q5f/jDH6b73ve+6YMf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nIj2notWrzlcfw4vrZl5Dc43X883Wl/6yfeczPca/MlPfpJOOeWUdJ3rXGf5v61+D6c8r/L/n9ftpRu3W4tXCUQ7d0Sr1xyuv05EO3dEq9e5rv4crnlOcux4AgKseD2bScX5wvcDH/hA9xWCfN+Mm9/85t3z5gAr35A73x9j6d5VOYTIXyU87LDDurDiHve4RxfG5N+mnXjiielv/uZvuseu/tdCxn4R0erNrz9azbne/LWUPBfynLjjHe/YtfEtb3lL98mp/NXBpfte5X+fP6mXv1L4la98JT3oQQ/q+p8/uXf44Yd3XyudxZyIVK85MZs5YV27svv6WK0f61r9c11eczfeeON073vfOz3gAQ/oWpnX2xxg5b/m+vjHP375fHvGGWd04z760Y9296fMfwAhh1l777139wc1bnGLW1ztFww15oVzR/05Eelcl+eDOVx/TjjXOdetvp5HW4drnIscczoCAqzp9HK0V5LvS/Wnf/qn6U53ulN38/UcSOWvIeTf5OaAKn8VMH+U/klPetLyc+abt+evITz5yU/uLozzT76Qzve6yo956EMfOlp9ax4oWr25/mg1583PLrvs0s2JCy+8MN3oRjfq7mGV58nXvva1tPPOO6fdd9+9u+fV0m/586eqlh6Tw638k7/ekjdR+SbCj33sY6vNiWj1mhP150S091y0es3h+nP4M5/5THdPq/zLg/x1/Isuuqi7d1X+a675E7AvfOELu7Aq/5Ig34cwfxIn33sy38w9j8+/VMo/Z511Vrr//e/ffWI6f1K25k+0tVi9dc/N5nD9dSLauSNavc519edwzXOSY09DQIA1jT6O9iouv/zy7kawj3rUo7obr+e/Evi3f/u33SdszjzzzO6vCD784Q9P17ve9bqvja3+FwRz0JUvrvfdd9+unvxVwfy1hJp/ZTBavdklYs35xv353lT5Jvz5ppof+tCHuhsD54vRRz/60d29rPINg/On8B7ykIcsz8f8SYBtttmmCzPzzdvzJir/Nazb3e52o83ZtR0oWr3mRN05Ec03Wr3Wtdmsa8961rO6r2XnTzznrwXm9XfXXXftPhmb/5n/IEb+d/mTWPkXTks/+RcN+QbZ+Tyef/mQg60cfuV7Vtb+ibYWq7fuudkcdq5bfc1xrqu/T8re0da12uclx48vIMCK38NRX0EOrHIgkUOKfCPY/JPvW5U/NXOTm9yku1F3vjF3/kRV/sTVPvvs092wPZ+E8l8lzBfR+VM4s/qJVm92iVZz7veDH/zgbmOU/7n0k/9i1fnnn999RTDPkfyJvPvc5z7pTW960/JXTrfffvuU/5fD0Fn9RKvXnKg/M6K956LVaw7Xn8PnnHNOdyP2Qw89tPsU9NJP/tRV/hRs/rpQDqTyp17z+Tj/4in/sZT885SnPCVttdVW3S8SZvkTbS1Wb93ZYQ7X9Y24DjvX1Z8T0da1+iKeYQoCAqwpdHHE15C/hpDvV5Q/YZM/hbV036p8kskXz/meGfkTWm94wxvShz/84e4vEeavF37kIx9J3/jGN7qvC9b+y4Krv9xo9ebao9Wc/1pg/uTdu9/97vSMZzwjLd2YP39FMH+6Lt/LIt/EP3+t9Oijj04///nPu6+15Pus5HnzsY99rLtn1qx+otVrTtSfGdHec9HqNYfrz+H8DFtssUV61ate1f2SaOkm7Pmm7fnTz/lTsPkPauT7C77uda/rviaYw6x8G4D8VwhzwJV/OTXLn2hrsXrrzw5zuK5xtHNHtHqd6+rOX0cnUCogwCqVmui4/FWC/JO/3pV/vv/973cXx/mCOH+SZuknj8s35M5fQ8h/bjvfDPDrX/9699vgX/7yl93XEvKN3fNveWv+RKs3W0SsefUb7uevmuRP2+UNU74/SP7J91XJXyPNQdXJJ5/cbZbyY84777zuq4X5E1n5RtE53MrhV+2faPWaE3XnRDTfaPVa12a7ruX1Lf/v5S9/efrSl76UTj/99Kutw69//eu7P6KRf6mQz+X5L/7mf5c/IZvX6vzV7ppf5V99fY+2Fqu37tl5ydccruMc7dwRrV7nutmc6+q8Oxx1ygICrCl3dy2vbenPKy8tykvB1bnnnrscNOSw4jvf+U7af//902Me85jlT2HlG77mUCvfiHvLLbfsjp4vSnKAtemmm1aRjFbvmq6r19+q8VIYtWYD82/G8v2r8k/u+/HHH99toPJ9znKAed3rXrfbTOWbAOff8Od7ZC39rP4n2seeGNHqNSdWzoCacyLaOhGtXnO4/hzOX/vLvwha8yd//ePud79796/zp57zefkFL3hB90c0rrjiirTRRht1vzjI6+/nP//57hPUSz9L6/TYa/DS8aKtxeqtuw6bw/XXiWjnjmj1OtfVn8O1zkeOu1gCAqwF6Xf+mkH+6kG+2M2/iX3Zy162/KfUTzrppO4TNvnCON/DKl8wL/0VwiOPPLK7aWz+yb/Rff/73999VTD/Ce+aP9HqzRbRas715pv95k1OvlfKXnvtlTbbbLOurf/yL//S3fcs//f99tsv5XtX5Pud5b9AeOKJJy7f/Pe9731vNy/y1wU333zzmlOi841UrzkxmzlhXav6trOuzWBdy+ffX/ziF936m7+Onf+SYP75t3/7t7RixYruk65vfvObuxuv568J5q/4569nL/0iKX+1Pz8u/3XXu93tbnUnxP+d6yKtxdHOHRHrNYfrX/8419Vd2lzD153Ddbvn6IsmIMBagI7nm2znr/9tt9123aes8r2KnvzkJ3e/wc035T7jjDO6T9jkG73mG3HnT2Xlv2qUv5KQf6ubA4qlr4vlC+ajjjqqqlq0ejNGtJq/9a1vdQFV/qro/e53v+7rn3l+5BAr35w9f1os9z9vnvKfYM9zIm+O8p9rzzfzz/dAy/eyyJuY/FXBHGzmT2TV+olWrzlRf05Ee89Fq9ccrj+Hc0CV7ymZ7xuZ/0pUPrfmP4ry/9s7E2grq/L/P4EDKE44ZGoqYGqpgLNLksTZUBNQEXFGEQmnTMQBUSM0HFsqmhYIhQM4pSFETjnlnAoqJEKiWDibuiSFfv/12f337eX4nnvfc+/Z73n3Pd+9Fiu799xzvufzPnt69vM8m7qS1K/ixlYOCdq3b++cWDRuAybN/+OPP3Y3/u64445uXCZtmxRvXhuyxTYWS2/YuVk2HH6ciG3uiE2v5rrwNhxyTtJ71ycBObDq4LmzGKZ21e233+6+LbWriKbBEcHPWPBSMyMZVUWhborBEpG11lprucgbTnZvvfVW22ijsNcsx6YXprFp5lZAoqxIB23btq2rfcYV7ETfPfXUUw03SyZTWggFZ/NECiH2wWnVZptt5mwidJ2r2PTKJsLbRGx9Lja9suHwNswhwaRJk2zmzJm2+uqr2z/+8Q9Xr4p0/Xnz5tm6667bcGlGcqlCJNYxxxzjDhooPM7B0uTJkxtuDg65rIltLJbesPUGZcPhx4nY5o7Y9GquC2/DIeckvXd9EpADqxU/d+pTUfOhb9++LkXsxhtvbPi2XLFN1BX1i0gRS+apJ+tELF682N555x1Xv2bXXXcNSis2vcCITbMvqMrpPs8WB5Zvjz76qEtD4aYqrlsvZxOkurCBIjqA6K2QLTa9sol8bELjWshep3Et9Ljmx1YiqIhsffHFFxse6IIFC1w09BZbbGFTpkxpqEHJC/wNsPw39a+IwuJwikja0C22sVh6w87NsmHNdaVjTmzrYa3Xwttw6HlJ71+/BOTAakXPHifTH//4RxchRTqYT+nae++9Xd0rboTzETU4H0g7ILyeKCwcXMuWLXML5tdee83dJBe6xaYXHrFppqgq9VJ4vtttt52LpqMNHjzYbXyuu+469zvakiVLnOOKNBZuFuzSpYuzCQoHT58+3caPH+9el3RsVdtGYtMrmwhvE7H1udj0yobD2zDjGlFV1LhiHO7cubMbOqlnRQ3KX/3qVw1OKMZX5mFSt5mfSeNmHL733ntdqjfRWqHHYd4/trFYesPPzbLhsIxjmzti06u5LvxcV+09gd5PBMoRkAOrldgGC2DqEXFrHA6o/v3724knnmg9evRwTgqKP5I6yKmubzg2fvazn7kaV9Q9YuGMY4Mb5aZNm+bqcoRqsemFQ2yaWWxSWJXUzzlz5rhoqTPPPNPVWsFpSU0VbhbcZ599Gh4zxdipg0V6iq+5QkrLmDFj3I2DOEZDtdj0yibC20RsfS42vbLh8DZMijXzKo4oUv64sff88893qdik6RP1yu2uXJziG5dmHHXUUa5G5dixY92P77zzTjv00EPt8ccfDx4NHdtYLL1h52bZcPhxIra5Iza9muvC23CovYHeVwTSCMiB1Qrs4q233nJF2XFYHXvsscaNRDfffLPNnz/fObNonTp1co4KImy4idA36m7wWtIMadRC8jcVhkITm144xKaZuij77ruvHXLIIc6xiVOS58xpP84saqtw0xXRepzqJ+uaUdeKAv9srGikkBIB4G+8CmEXsemVTYS3idj6XGx6ZcPhbZgbXomA7tmzp1100UXuwhQODbjdlyjXXXbZxc29pGVfeumly6Vk8zff//733eEB7dNPP3X/uDwjZIttLJbesHOzbDj8OBHb3BGbXs114W045Jyk9xYBObBaqQ2wIOb0lgKwvhA70Vac1hJZRUrY1KlTXVQWta+OOOIId6scp7y+3hE30uXVYtMLl9g0k0pKHRWcmN/6/9fAU4gfJyapgXfffbcr1r7ffvvZWWedZcOHD3c3TVJThag9foYzNK8Wm17ZRHjLiK3PxaZX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W54+QQREoDEC0TmwSA0kdfD555933+v999+3uXPn2i677NJQvL1Ijzw2vbCLTbP0hrd4MQ7LWHzD8tW4Jr5pBNTvwtqF+Iblq3FNfDWuhbcBfYIIiEDxCETjwPK3Cx5//PG20kor2Q033OBuKKT21VlnnWWjR492aYNFabHphVtsmqU3vLWLcVjG4huWr8Y18U0joH4X1i7ENyxfjWviq3EtvA3oE0RABIpLIBoHFgipqUD6IEXZX375ZeMa5Ouuu8569+5dSMKx6RXj8GYkmxDjUgKx2URsejWuqc+lEYjNjqU3rB3HxlfjWlh7iJFvjJpj7HfhLU+fIAIi0BSBqBxYc+bMcbWt1l57baNgO5FXRW6x6YVlbJqlN3wPEOOwjMU3LF+Na+KbRkD9LqxdiG9YvhrXxFfjWngb0CeIgAgUk0BUDiwQEnE1aNAga9euXTGJlqiKTa8Yhzcr2YQYlxKIzSZi06txTX0ujUBsdiy9Ye04Nr4a18LaQ4x8Y9QcY78Lb3n6BBEQgcYIROfA0uMUAREQAREQAREQAREQAREQAREQAREQARGoLwJyYNXX89a3FQEREAEREAEREAEREAEREAEREAEREIHoCMiBFd0jk2AREAEREAEREAEREAEREAEREAEREAERqC8CcmDV1/PWtxUBERABERABERABERABERABERABERCB6AjIgRXdI5NgERABERABERABERABERABERABERABEagvAnJg1dfz1rcVAREQAREQAREQAREQAREQAREQAREQgegIyIEV3SOTYBEQAREQAREQAREQAREQAREQAREQARGoLwJyYNXX89a3FQEREAEREAEREAEREAEREAEREAEREIHoCMiBFd0jk2AREAEREAEREAEREAEREAEREAEREAERqC8CcmDV1/PWtxUBERABERABERABERABERABERABERCB6AjIgRXdI5NgERABERABERCBohM49thjbeLEiU7mCiusYB07drSuXbvagAEDjN+1adMm01e4+eab7fTTT7ePP/440+v1IhEQAREQAREQARForQTkwGqtT1bfSwREQAREQAREoGYEcFItXrzYJkyYYMuWLXP/PWPGDLvkkktst912s3vvvdc5tppqcmA1RUi/FwEREAEREAERqBcCcmDVy5PW9xQBERABERABEciNAA4soqbuueee5T7zoYcesj333NNuuukmO+GEE+zKK690Tq758+e7KK0DDzzQxo4dax06dLBHHnnEevXqtdzfjxo1yi688EL797//beedd57deuut7nO23npr+8UvfmG77757bt9RHyQCIiACIiACIiACeRKQAytP2vosERABERABERCBuiBQzoHFl+/evbttsMEGdv/999vVV19t3bp1s06dOjkn1tChQ22PPfawcePG2ZdffmnXX3+9XXDBBTZ37lzHDccW/0488UR79dVX7dJLL3Xvdffdd9v5559vs2bNsu985zt1wVhfUgREQAREQAREoL4IyIFVX89b31YEREAEREAERCAHAo05sA4//HB7+eWXnQOqtN1xxx02ZMgQe//9992v0lIIFy5caJ07dzb+F+eVb3vttZfttNNONmbMmBy+oT5CBERABERABERABPIlIAdWvrz1aSIgAiIgAiIgAnVAoDEHVv/+/W327Nn2yiuv2AMPPODqYs2ZM8f+9a9/2dKlS23JkiX2+eef2yqrrJLqwJo2bZodcMABtuqqqy5HkrTCvn372u23314HhPUVRUAEREAEREAE6o2AHFj19sT1fUVABERABERABIITaMyBxW2EG2+8sV177bW25ZZb2sknn2w4taiB9fjjj9ugQYPso48+sjXXXDPVgYWDauDAgc4B1rZt2+W+C+mF66+/fvDvpw8QAREQAREQAREQgbwJyIGVN3F9ngiIgAiIxkBZzQAAAmVJREFUgAiIQKsn0FQR9/Hjx9vqq69uAwYMcBFXbdq0cUxGjx5tI0eObHBg3XLLLXbSSSfZp59+2sDsb3/7m22xxRb26KOPuhsN1URABERABERABESgHgjIgVUPT1nfUQREQAREQAREIFcCOLAWL17sbhhctmyZ++8ZM2a4dEFuCuR2QtIIKehOIXduH3ziiSfsnHPOsUWLFjU4sJ588knr0aOHSzWk2Dtphfw78sgj3euvuOIK23bbbe29996zBx980Iju6t27d67fVR8mAiIgAiIgAiIgAnkQkAMrD8r6DBEQAREQAREQgboigANr4sSJ7juvsMIKttZaazkH1BFHHGHHHHNMQ8TVVVddZZdddpl9/PHH1rNnT5caePTRRzc4sPh7UgynTp1qH3zwgY0aNcouvPBC++qrr1y01qRJk5zDa5111rFddtnFLrroIttmm23qirW+rAiIgAiIgAiIQH0QkAOrPp6zvqUIiIAIiIAIiIAIiIAIiIAIiIAIiIAIREtADqxoH52Ei4AIiIAIiIAIiIAIiIAIiIAIiIAIiEB9EJADqz6es76lCIiACIiACIiACIiACIiACIiACIiACERLQA6saB+dhIuACIiACIiACIiACIiACIiACIiACIhAfRCQA6s+nrO+pQiIgAiIgAiIgAiIgAiIgAiIgAiIgAhES0AOrGgfnYSLgAiIgAiIgAiIgAiIgAiIgAiIgAiIQH0QkAOrPp6zvqUIiIAIiIAIiIAIiIAIiIAIiIAIiIAIREtADqxoH52Ei4AIiIAIiIAIiIAIiIAIiIAIiIAIiEB9EPh/D9xpklROZ+sAAAAASUVORK5CYII=" width="1200">





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
