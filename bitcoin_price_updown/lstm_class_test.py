import pandas as pd
import talib
#nltk.download('punkt')
#notclean = pd.read_csv("../twitterdata/tweet_t.csv"  , usecols=['dt', 'text' , 'vader' , 'polarity' ,'sensitivity'])
notclean = pd.read_csv("tweet_t.csv"  , usecols=['dt', 'text' , 'vader' , 'polarity' ,'sensitivity'])
notclean =  notclean.dropna()

notclean  = notclean.drop_duplicates(subset=['text'])





minute = "60min"
notclean['dt'] = pd.to_datetime(notclean['dt'] , errors='coerce', format='%Y-%m-%d %H:%M:%S')
notclean['dt'] = notclean['dt'].dt.strftime("%Y-%m-%d %H:%M:%S")
notclean['dt'] = pd.to_datetime(notclean['dt'])

notclean['DateTime'] = notclean['dt'].dt.floor('min')
print(notclean.head(2))
vdf = notclean.groupby(pd.Grouper(key='dt',freq=minute)).size().reset_index(name='tweet_vol')

vdf.index = pd.to_datetime(vdf.index)
vdf=vdf.set_index('dt')

notclean.index = pd.to_datetime(notclean.index)

vdf['tweet_vol'] =vdf['tweet_vol'].astype(float)

df = notclean.groupby('DateTime').agg(lambda x: x.mean())

df['tweet_vol'] = vdf['tweet_vol']


df = df.drop(df.index[0])

df = df.dropna()

########################## lstm model start #############################
##########   get bitcoin price from 14 days before to now            ##################

import datetime

start = vdf.head(1).index[0] - datetime.timedelta( 14 )
end = vdf.tail(1).index[0] + datetime.timedelta()

#start = (datetime.date.today() - datetime.timedelta( NUM_DAYS ) )

import talib
import cryptocompare


pricelimit = cryptocompare.get_historical_price_hour('BTC', 'USD' , limit = 341, toTs=datetime.datetime(end.year,end.month,end.day,end.hour))

simple_list = []
for i in pricelimit:    

    simple_list.append([i.get("time"), i.get("close")])
  

control=pd.DataFrame(simple_list,columns=['Timestamp','Close'])
    
control["Timestamp"] = control["Timestamp"] + 10800
control["DateTime"] = pd.to_datetime(control["Timestamp"], unit='s')

control['DateTime'] = control['DateTime'].dt.floor('min')
control =  control.drop(["Timestamp"] , axis = 1)
control = control.set_index("DateTime")
control["ma"] = talib.MA(control['Close'], timeperiod=5)
control["diff"] = control['Close'] - control["ma"]

def buysell_signal(diff):
    
    
    if diff > 0:
        return 1
    else:
        return 0
    
control["diff"] = control["diff"].apply(buysell_signal) 
    

Final_df = pd.merge(df,control, how='inner',left_index=True, right_index=True)

Final_df =Final_df.drop(['dt', 'ma' ], axis=1)

Final_df.columns = ['vader','polarity','sensitivity','tweet_vol', 'Close', 'diff' ]

Final_df = Final_df[['vader','polarity','sensitivity','tweet_vol', 'Close', 'diff']]



data = Final_df
x = data.filter(["polarity",'Close', 'tweet_vol', 'diff'])


from sklearn import preprocessing

xx = x.values 
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(xx)
#df = pd.DataFrame(x_scaled)
print(x_scaled.shape)

from sklearn.preprocessing import LabelEncoder



# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
# input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (df.columns[j], i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (df.columns[j])) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (df.columns[j], i)) for j in range(n_vars)]
# put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg
 
# integer encode direction
encoder = LabelEncoder()
xx[:,3] = encoder.fit_transform(xx[:,3])
# ensure all data is float
xx = xx.astype('float32')
# normalize features
scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(xx)
# frame as supervised learning
reframed = series_to_supervised(xx, 2, 1)
# drop columns we don't want to predict
reframed.drop(reframed.columns[[8,9,10]], axis=1, inplace=True)
print(reframed.head(1))

hours_move = xx

hours_move = series_to_supervised(hours_move,2,1)



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dropout,Activation
from tensorflow.keras.layers import Dense

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from keras import metrics as met2
import math





labels = reframed.values[:,-1]
print(reframed.shape)
features = reframed.values[:,:8]
print(features.shape,labels.shape)
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.5 ,shuffle = False )

x_train = x_train.reshape((x_train.shape[0], 2, 4))
x_test = x_test.reshape((x_test.shape[0],2, 4))
print(x_train.shape)
print(x_test.shape)



model = Sequential()
model.add(Bidirectional(LSTM((4), input_shape=(x_train.shape[1], x_train.shape[2]))))
model.add(Dropout(0))
model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(loss='binary_crossentropy', metrics=[met2.binary_accuracy], optimizer='adam')
history = model.fit(x_train, y_train, epochs=50, batch_size=10, validation_data=(x_test, y_test), verbose=2)
print(model.summary())


# make a prediction

yhat = model.predict(x_test)
rmse = math.sqrt(mean_squared_error(y_test, yhat))
print('Test RMSE: %.3f' % rmse)


yhat = yhat.astype("float64")
y2 = yhat
for i in range(yhat.shape[0]):
    if y2[i] >0.5:
        y2[i] = 1
    else:
        y2[i] = 0
matrix = metrics.confusion_matrix(y_test,yhat)
print(matrix)

score = model.evaluate(x_test, y_test, batch_size=72, verbose=1)
print('Test score:', score[1])


########################## save all test data result+time ##############################
print(yhat)
pred_time = x.tail(len(yhat))
pred_time["diff_pre"] = yhat
pred_time['Datetime'] = df.tail(len(yhat)).index
pred_time["Future_time"] = pred_time['Datetime'] + datetime.timedelta(hours = 1)

pred_time = pred_time.filter(["diff",'Datetime', 'Future_time' , 'Close'])


pred_time.to_csv(r'lstm_class_test.csv', index = False, header=True)





# pred_t


############################ real time classficiation data prep  ###################################

import numpy as np
next_pre = np.concatenate((scaled[-2,:],scaled[-1,:]))
next_pre = next_pre.reshape(1, next_pre.shape[0])
next_pre  = next_pre.reshape((next_pre.shape[0], 2, 4))

tmp = 0    
for i in range(0,3):
    new_pre = model.predict(next_pre)
    if new_pre[0][0] > 0.5:
        new_pre[0][0] = 1
        tmp = tmp + new_pre[0][0]
    else:
        new_pre[0][0] = 0        
        tmp = tmp + new_pre[0][0]
if tmp > 1.1:
    new_pre[0][0] = 1        
else:
    new_pre[0][0] = 0     

########################## real time classfication data result ##############################
import datetime        
time =  x.tail(1).index[0] + datetime.timedelta(hours = 1)

print("------------------------------------")
print(new_pre[0][0] ,"signal time for " , time)
print("------------------------------------")

lstm_class = new_pre[0][0]

