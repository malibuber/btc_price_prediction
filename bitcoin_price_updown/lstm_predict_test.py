###########################################################################################
################################ lstm price prediction ####################################
### son y_test sayısı kadar olucak ###
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense


from sklearn.metrics import mean_squared_error

import pandas as pd
import talib
#nltk.download('punkt')
#notclean = pd.read_csv("../twitterdata/tweet_t.csv"  , usecols=['dt', 'text' , 'vader' , 'polarity' ,'sensitivity'])
notclean = pd.read_csv("tweet_t.csv"  , usecols=['dt', 'text' , 'vader' , 'polarity' ,'sensitivity'])
notclean =  notclean.dropna()

notclean  = notclean.drop_duplicates(subset=['text'])






notclean['dt'] = pd.to_datetime(notclean['dt'] , errors='coerce', format='%Y-%m-%d %H:%M:%S')
notclean['dt'] = notclean['dt'].dt.strftime("%Y-%m-%d %H:%M:%S")
notclean['dt'] = pd.to_datetime(notclean['dt'])

notclean['DateTime'] = notclean['dt'].dt.floor('min')
print(notclean.head(2))

minute = "60min"


vdf = notclean.groupby(pd.Grouper(key='dt',freq=minute)).size().reset_index(name='tweet_vol')

vdf.index = pd.to_datetime(vdf.index)
vdf=vdf.set_index('dt')

notclean.index = pd.to_datetime(notclean.index)

vdf['tweet_vol'] =vdf['tweet_vol'].astype(float)

df = notclean.groupby('DateTime').agg(lambda x: x.mean())

df['Tweet_vol'] = vdf['tweet_vol']

df = df.drop(df.index[0])

import datetime

start = vdf.head(1).index[0] - datetime.timedelta( 14 )
end = vdf.tail(1).index[0] + datetime.timedelta()

#start = (datetime.date.today() - datetime.timedelta( NUM_DAYS ) )


import cryptocompare

# get bitcoin price minu
#pricelimit = cryptocompare.get_historical_price_minute('BTC', 'USD', toTs=datetime.datetime.now())
# get bitcoin price hourly
pricelimit = cryptocompare.get_historical_price_hour('BTC', 'USD' , limit = 341, toTs=datetime.datetime(end.year,end.month,end.day,end.hour))

simple_list = []
for i in pricelimit:    

    simple_list.append([i.get("time"), i.get("close")])
  
    #df2 = pd.DataFrame({"Timestamp": time, "Close": close}) 
    #print(df2)
control=pd.DataFrame(simple_list,columns=['Timestamp','Close'])
#df1.append(df2, ignore_index = True)     
control["Timestamp"] = control["Timestamp"] + 10800
control["DateTime"] = pd.to_datetime(control["Timestamp"], unit='s')

control['DateTime'] = control['DateTime'].dt.floor('min')
control =  control.drop(["Timestamp"] , axis = 1)
control = control.set_index("DateTime")
control["ma"] = control["Close"] - control["Close"].shift()
control["ma"] = talib.MA(control['ma'], timeperiod=5)
#control["ma"] = talib.MA(control['Close'], timeperiod=5)
last_price = control["Close"].tail(1).values[0]




#control = control.dropna()

##newDf['Volume_BTC'] = newDf['Volume_BTC'] - newDf['Volume_BTC'].shift()
##newDf['Volume_BTC'] = talib.MA(newDf['Volume_BTC'], timeperiod=5).shift()

Final_df = pd.merge(df,control, how='inner',left_index=True, right_index=True)

Final_df =Final_df.drop(['dt'], axis=1)

Final_df.columns = ['Polarity', 'vader', 'Sensitivity','Tweet_vol', 'Close', 'Volume_BTC' ]

Final_df = Final_df[['Polarity', 'vader','Sensitivity','Tweet_vol','Volume_BTC', 'Close']]

df = Final_df

df = df[['Close', 'Polarity', 'vader', 'Sensitivity','Tweet_vol', 'Volume_BTC']]


df = df.dropna()

from pandas import DataFrame
from pandas import concat
def series_to_supervised2(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg


from sklearn.preprocessing import MinMaxScaler

values = df.values
cols = df.columns.tolist()
cols = cols[-1:] + cols[:-1]
df = df[cols]
df = df[['Close', 'Polarity', 'vader', 'Sensitivity','Tweet_vol', 'Volume_BTC']]
#ma_result MA(df.Close_Price, timeperiod=10, matype=0)
#df =df.drop(["vader_sent"], axis=1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(df.values)


n_hours = 3 #adding 3 hours lags creating number of observations 
n_features = 6 #Features in the dataset.
n_obs = n_hours*n_features
n_feat = 5

reframed = series_to_supervised2(scaled, n_hours, 1)
reframed.head()
#reframed.drop(reframed.columns[-n_feat], axis=1)
reframed = reframed.drop(reframed.columns[[-5 ,-4 ,-3 ,-2 ,-1] ], axis=1)
reframed.head()



from sklearn.model_selection import train_test_split



values = reframed.values
features = values[: ,:-1] 
labels = values[:, -1:]
#n_train_hours = round(len(values)*(0.66))
#train = values[:n_train_hours, :]
#test = values[n_train_hours:, :]
train_X, test_X, train_y, test_y= train_test_split(features, labels, test_size=0.33 , shuffle = False)
#train_X, train_y = train[:, :-1], train[:, -1:]
#test_X, test_y = test[:, :-1], test[:, -1:]

train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

from numpy import concatenate
from math import sqrt

model = Sequential()
model.add(LSTM(50,return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=100, batch_size=10, validation_data=(test_X, test_y), verbose=2, shuffle=False,validation_split=0.2)
print(test_X.shape)
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], n_hours* n_features,))
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, -n_feat:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, -n_feat:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# calculate RMSE
mse = (mean_squared_error(inv_y, inv_yhat))
print('Test MSE: %.3f' % mse)
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)



caldata =  pd.DataFrame(data = inv_y , columns=['result'])
caldata = caldata.drop(caldata.index[0])
inv_yhat1 = pd.DataFrame(data =inv_yhat , columns=["result"]) 
inv_y1 = pd.DataFrame(data =inv_y , columns=["result"])
inv_yhat1.result = inv_yhat1.result.shift() - inv_y1.result
#inv_yhat1.result = inv_yhat1.result - inv_y1.result.shift()
inv_y1.result = inv_y1.result.shift() - inv_y1.result
#inv_y1.result = inv_y1.result - inv_y1.result.shift()

inv_yhat1.dropna()
inv_y1.dropna()

def updown(score):
    
    if (score > 0):
        return 1
    else:
        return 0

inv_y1.result =inv_y1.result.apply(updown)
inv_yhat1.result =inv_yhat1.result.apply(updown)
  
def buysell(real , pre ):
    cal =  pd.DataFrame(data = inv_y , columns=['result'])
    cal = cal.drop(cal.index[0])

    prev = 0
    realv = 0
    for i ,r in real.iterrows():
        
        if r.result == 1:
            realv = realv + cal.result[i]
        elif r.result == 0:
            realv = realv - cal.result[i]
    for i ,r in pre.iterrows():
    
        if r.result == 1:
            prev = prev + cal.result[i]
        elif r.result == 0:
            prev = prev - cal.result[i]
    
    return prev , realv

inv_y1 = inv_y1.drop(inv_y1.index[0])
inv_yhat1 = inv_yhat1.drop(inv_yhat1.index[0])
resultpre , resultreal = buysell(inv_y1 , inv_yhat1 )

print(resultpre ," ----" ,resultreal , "-------" , resultreal - resultpre)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(inv_y1, inv_yhat1))

tn, fp, fn, tp = confusion_matrix(inv_y1, inv_yhat1).ravel()

score_pre = ((tn+tp)/(tn+fp+fn+tp))

print((tn+tp)/(tn+fp+fn+tp))

print(inv_yhat1.result)
print(len(inv_yhat1.result))
########################## save all test data result+time ##############################
pred_time = Final_df.tail(len(inv_yhat1.result))

pred_time['Datetime'] = Final_df.tail(len(inv_yhat1.result)).index
pred_time["Future_time"] = pred_time['Datetime'] + datetime.timedelta(hours = 1)
inv_yhat1 = inv_yhat1.to_numpy()
pred_time["diff_pre"] = inv_yhat1
pred_time = pred_time.filter(['Datetime' ,'Future_time' ,'Close',"diff_pre"])


pred_time.to_csv(r'lstm_predict_test.csv', index = False, header=True)


'''
     
import numpy as np
next_pre = np.concatenate((scaled[-3,:] ,scaled[-2,:],scaled[-1,:]))
next_pre = next_pre.reshape(1, next_pre.shape[0])
next_pre  = next_pre.reshape((next_pre.shape[0], n_hours, n_features))
new_pre = model.predict(next_pre)
next_pre = next_pre.reshape((next_pre.shape[0], n_hours* n_features,))
result = concatenate((new_pre, next_pre[:, -n_feat:]), axis=1)
result = scaler.inverse_transform(result)
result = result[:,0]

print(result)


########################## real time classfication data result ##############################





import datetime
if result  > last_price:
    signal = 1
else:
    signal = 0
    
time =  Final_df.tail(1).index[0] + datetime.timedelta(hours = 1)
print("------------------------------------")
print(signal ,"signal time for " , time)
print("------------------------------------")

lstm_predict = signal

'''