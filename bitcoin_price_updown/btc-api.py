

from binance.client import Client

api_key = "7tRYMJfc4aPLUaSg6gIDWHuUARn9yHJlbLpS3s2ohwPd6VXOvV9wjNBc0ZUlVpuP "

api_secret = "cUtxXVMEiPU3m9iqvIOAKXCgP3TSTXrCrwpP22693jjrgm5tohFP6m6cujuskXoA"

client = Client(api_key, api_secret)
info = client.get_symbol_info('BTCUSDS')
#candles = client.get_klines(symbol='BTCUSDS', interval=Client.KLINE_INTERVAL_1HOUR , "1 day ago UTC")

klines = client.get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_1HOUR, "1 day ago GMT+03:00")



import pandas as pd
data = pd.DataFrame()
simple_list = []
for i in klines:    
    simple_list.append([i[0], i[4] , i[3] ,i[2] ,i[1] , i[5]])
    #simple_list.append([i.get("time"), i.get("close"), i.get("low"), i.get("high"), i.get("open") , i.get("volumefrom") ])
  
    #df2 = pd.DataFrame({"Timestamp": time, "Close": close}) 
    #print(df2)
data=pd.DataFrame(simple_list,columns=['Timestamp','close','low' ,'high','open','volume'])
#df1.append(df2, ignore_index = True)     

data["DateTime"] = pd.to_datetime(data["Timestamp"], unit='ms')

data['DateTime'] = data['DateTime'].dt.floor('min')
data =  data.drop(["Timestamp"] , axis = 1)
data = data.set_index(pd.DatetimeIndex(data['DateTime']))

last_price = data["close"].tail(1).values[0]



import pandas as pd
import cryptocompare
import datetime
data = pd.DataFrame()
simple_list = []
# get bitcoin price minu
#pricelimit = cryptocompare.get_historical_price_minute('BTC', 'USD', toTs=datetime.datetime.now())
# get bitcoin price hourly
#pricelimit = cryptocompare.get_historical_price_hour('BTC', 'USD' , limit = 341, toTs=datetime.datetime(end.year,end.month,end.day,end.hour))
pricelimit = cryptocompare.get_historical_price_hour('BTC', 'USD' , limit = 100, toTs=datetime.datetime.now())
for i in pricelimit:    
    
        simple_list.append([i.get("time"), i.get("close"), i.get("low"), i.get("high"), i.get("open") , i.get("volumefrom") ])
      
        #df2 = pd.DataFrame({"Timestamp": time, "Close": close}) 
        #print(df2)
data=pd.DataFrame(simple_list,columns=['Timestamp','close','low' ,'high','open','volume'])
#df1.append(df2, ignore_index = True)     
data["Timestamp"] = data["Timestamp"] + 10800
data["DateTime"] = pd.to_datetime(data["Timestamp"], unit='s')


