###########################################################################
########### lstm prediction using just talib no sentiment #################
import datetime
import pandas as pd
import talib
from finta import TA
import numpy as np
INDICATORS = ['RSI', 'MACD', 'STOCH','ADL', 'ATR', 'MOM', 'MFI', 'ROC', 'OBV', 'CCI', 'EMV', 'VORTEX']


data = pd.DataFrame()
import cryptocompare

# get bitcoin price minu
#pricelimit = cryptocompare.get_historical_price_minute('BTC', 'USD', toTs=datetime.datetime.now())
# get bitcoin price hourly
#pricelimit = cryptocompare.get_historical_price_hour('BTC', 'USD' , limit = 341, toTs=datetime.datetime(end.year,end.month,end.day,end.hour))
pricelimit = cryptocompare.get_historical_price_hour('BTC', 'USD' , limit = 341, toTs=datetime.datetime.now())
simple_list = []
for i in pricelimit:    

    simple_list.append([i.get("time"), i.get("close"), i.get("low"), i.get("high"), i.get("open") , i.get("volumefrom") ])
  
    #df2 = pd.DataFrame({"Timestamp": time, "Close": close}) 
    #print(df2)
data=pd.DataFrame(simple_list,columns=['Timestamp','close','low' ,'high','open','volume'])
#df1.append(df2, ignore_index = True)     
data["Timestamp"] = data["Timestamp"] + 10800
data["DateTime"] = pd.to_datetime(data["Timestamp"], unit='s')
data['DateTime'] = data['DateTime'].dt.floor('min')
time_all = data["DateTime"]
data =  data.drop(["Timestamp"] , axis = 1)
data = data.set_index(pd.DatetimeIndex(data['DateTime']))
control_data = data["close"]
last_price = data["close"].tail(1).values[0]
print(data.shape)

def _exponential_smooth(data, alpha):


    return data.ewm(alpha=alpha).mean()


data = _exponential_smooth(data, 0.65)


def _get_indicator_data(data):
    #import talib
    from finta import TA
    for indicator in INDICATORS:
        ind_data = eval('TA.' + indicator + '(data)')
        if not isinstance(ind_data, pd.DataFrame):
            ind_data = ind_data.to_frame()
        data = data.merge(ind_data, left_index=True, right_index=True)
    data.rename(columns={"14 period EMV.": '14 period EMV'}, inplace=True)

    # Also calculate moving averages for features
    data['ema50'] = data['close'] / data['close'].ewm(50).mean()
    data['ema21'] = data['close'] / data['close'].ewm(21).mean()
    data['ema15'] = data['close'] / data['close'].ewm(14).mean()
    data['ema5'] = data['close'] / data['close'].ewm(5).mean()

    # Instead of using the actual volume value (which changes over time), we normalize it with a moving volume average
    data['normVol'] = data['volume'] / data['volume'].ewm(5).mean()

    # Remove columns that won't be used as features
    del (data['open'])
    del (data['high'])
    del (data['low'])
    del (data['volume'])
    
    
    return data

data = _get_indicator_data(data)
data = data.dropna()

def _produce_prediction(data, window):

    prediction = (data['close'] >= data['close'].shift(window))
    prediction = prediction.iloc[:-window]
    data['pred'] = prediction.astype(int)
    
    return data

data = _produce_prediction(data, window=1)
close_price = data['close'].tail(1).values[0]
del (data['close'])
#data = data.dropna() # Some indicators produce NaN values for the first few rows, we just remove them here
data.tail()
precdict_time =  data.tail(1).index[0]
time1 =  data.tail(1).index[0] + datetime.timedelta(hours = 1)
print(time1)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dropout,Activation
from tensorflow.keras.layers import Dense

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from keras import metrics as met2
import math


def _train_random_forest(X_train, y_train, X_test, y_test):

    """
    Function that uses random forest classifier to train the model
    :return:
    """
    
    model = Sequential()
    #model.add(Conv2D(100, 2, activation='relu'))
    #model.add(SimpleRNN(32))
    model.add(Bidirectional(LSTM((50), input_shape=(X_train[0], X_train[1]))))
    #model.add(ConvLSTM2D(2,kernel_size = (3,3),input_shape =(None,x_train.shape[1], x_train.shape[2],1) ,padding='same', return_sequences=True))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.add(Activation("sigmoid"))
    
    model.compile(loss='binary_crossentropy', metrics=[met2.binary_accuracy], optimizer='adam')
    # fit network
    history = model.fit(X_train, y_train.values, epochs=50, batch_size=10, validation_data=(X_test, y_test.values), verbose=2)
    # plot history
    print(model.summary())
    
    
    # make a prediction
    
    #X_test = X_test .reshape((X_test.shape[0], X_test.shape[2]))
    yhat = model.predict(X_test)
    #print(yhat)
    #print(yhat.score)
    rmse = math.sqrt(mean_squared_error(y_test, yhat))
    print('Test RMSE: %.3f' % rmse)

    return yhat,model

def updown(result):
    
    for i in range(0, len(result)):
        if result[i] > 0.5:
            result[i] = 1
        else:
            result[i] = 0
            
            
    return result
# Split data into equal partitions of size len_train
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0, 1) )
data = scaler.fit_transform(data)
data = pd.DataFrame(data=data)

real_pre_input =  data.tail(1)
real_pre_input =  real_pre_input.drop([19], axis=1)
data = data.dropna()
num_train = 10 # Increment of how many starting points (len(data) / num_train  =  number of train-test sets)
len_train = 40 # Length of each train-test set

# Lists to store the results from each model
rf_RESULTS = []
#data = data.tail(60)
rf_all = []
i = 0
while True:
    # Partition the data into chunks of size len_train every num_train days
    df = data.iloc[i * num_train : (i * num_train) + len_train ]
    i += 1
    print(i * num_train, (i * num_train) + len_train)
    if len(df) < 40:
        break
    y = df[19]
    features = [x for x in df.columns if x not in [19]]
    X = df[features]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.33 ,shuffle=False)
    #(28, 19)
    X_train = X_train.values.reshape(X_train.shape[0] , 1 , X_train.shape[1])
    X_test = X_test.values.reshape(X_test.shape[0] , 1 , X_test.shape[1])
    rf_model , model = _train_random_forest(X_train, y_train, X_test, y_test)
    
    # 0.5 yukari asagi 
    rf_prediction = updown(rf_model)
    rf_prediction = rf_prediction.flatten()

    from sklearn.metrics import accuracy_score
    rf_all.append(rf_prediction)

    rf_accuracy = accuracy_score(y_test.values, rf_prediction)
    rf_RESULTS.append(rf_accuracy)
    print(rf_accuracy)

real_pre_input = real_pre_input.values.reshape(real_pre_input.shape[0] , 1 , real_pre_input.shape[1])

tmp = 0    
for i in range(0,3):
    pre_future = model.predict(real_pre_input)
    if pre_future[0][0] > 0.5:
        pre_future[0][0] = 1
        tmp = tmp + pre_future[0][0]
    else:
        pre_future[0][0] = 0        
        tmp = tmp + pre_future[0][0]
if tmp > 1.1:
    pre_future[0][0] = 1       
else:
    pre_future[0][0] = 0        

print("------------------------------------")
print((pre_future[0][0]) , "time pre is " , time1)
print("------------------------------------")

import numpy as np
pred_time = pd.DataFrame()
time_all = time_all.to_frame()
time_all = time_all.reset_index()



rf_alls = []
for i in rf_all:
    for j in i:
       rf_alls.append(j) 
print(rf_all)
print(len(rf_alls))
control_data = control_data.reset_index()
print(control_data.tail(len(rf_alls)))
rf_all = np.array(rf_alls) 
pred_time['Datetime'] = time_all["DateTime"].tail(len(rf_alls))
pred_time["Future_time"] = pred_time['Datetime'] + datetime.timedelta(hours = 1)
pred_time["close"] = control_data["close"].tail(len(rf_alls))
print(pred_time)
pred_time.to_csv(r'lstm_prep_talib.csv', index = False, header=True)
print(len(pred_time))
pred_time["diff"] = rf_all
pred_time.to_csv(r'lstm_prep_talib.csv', index = False, header=True)


