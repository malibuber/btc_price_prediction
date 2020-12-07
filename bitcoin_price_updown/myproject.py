# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 17:26:15 2020

@author: mehmet
"""
import pandas as pd
from flask import Flask ,render_template
app = Flask(__name__)



def buysell(signal):
    
    if signal > 0:
        return "buy"
    elif signal == 0:
        return "sell"

####  Qwerty12345-.,

@app.route('/',methods = ['POST' , 'GET'])
def index():

    df = pd.read_csv("result.csv")
    
    #df = df.tail(10)
    df = df.sort_index(ascending=False)
    df["lstm_class"] = df["lstm_class"].apply(buysell)
    df["lstm_predic"] = df["lstm_predic"].apply(buysell)
    df["lstm_price"] = df["lstm_price"].apply(buysell)
    df["signal1"] = df["signal1"].apply(buysell)
    df["signal2"] = df["signal2"].apply(buysell)
    df["signal3"] = df["signal3"].apply(buysell)
    e_list = []
    for i , r in df.iterrows():
        
        e_list.append([r["lstm_class"],r["lstm_predic"],r["lstm_price"],r["predict_time"] ,r["future_time"],r["time_price"] ,
                       r["future_price"], r["signal1"], r["signal2"],r["signal3"] ])
        
    lis = len(e_list)
    return render_template('index.html' , my_list = e_list , lis = lis)
import time
def sleep():
    print("sleep")
    time.sleep(5)
    print("sleep---")
#from threading import Thread
#import untitled4   
#Thread(target=untitled4.precstart).start()

if __name__ == "__main__":
    # http://127.0.0.1:80/
    app.run(port=80)
    