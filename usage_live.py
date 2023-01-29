# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 02:43:48 2022

@author: Administrator
"""

import threading as th
from flask import Flask, request, jsonify, render_template
import pandas as pd
import json



app = Flask(__name__)


@app.route('/')
def welcome():
    return "Hi Trader!!"


@app.route('/webhook', methods=['POST'])
def webhook():
    global data
    global ack
    data = json.loads(request.data)
    data = pd.DataFrame(data)
    data['Time'] = (pd.to_datetime(data['Time'], unit='ms'))
    ack = True

    return 'complete'



def run_rec(app):
    app.run(host='0.0.0.0', port=5000)


if __name__ == '__main__':
    rec_thread = th.Thread(target=run_rec, args=(app,))
    rec_thread.start()
    data = {}
    ack = False
    input_size = 2
    sequence_length = 10
    num_layers = 2
    hidden_size = 8
    while (True):
        if ack == True:
            # do some thing like :

            # input_data = torch.rand(1, sequence_length, input_size)
            # nn = Dummy_model(input_size, hidden_size, num_layers)
            # nn(input_data)

            # or use the original data
            print(data)
            print(data['Open'])
            print(data['Close'])
            print(data['High'])
            print(data['Low'])
            # you can do whatever you want with data
            #
            # (*** VERY IMPORTANT ***) then make ack False
            ack = False
