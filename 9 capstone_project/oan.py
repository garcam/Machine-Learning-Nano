# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal
"""

import oandapy
import pandas as pd
#pd.set_option('precision',10)
oanda = oandapy.API(environment="practice", access_token="11b9647ffb3cb6c74f1d4a85b3ca5651-a6cde9a94ae26badf549f3d92d7f74c3")
#Get price for an instrument
response = oanda.get_prices(instruments="EUR_USD,AUD_USD,GBP_USD")
#prices = response.get("prices")
#asking_price = prices[0].get("ask")
prices_table=pd.DataFrame.from_records(response.get("prices"))




import json
from oandapyV20 import API    # the client
import oandapyV20.endpoints.trades as trades

access_token = "11b9647ffb3cb6c74f1d4a85b3ca5651-a6cde9a94ae26badf549f3d92d7f74c3"
accountID = "101-004-6446981-001"
client = API(access_token=access_token)

# request trades list CON ESTO TENGO MI OPEN POSITIONS
r = trades.TradesList(accountID)
rv = client.request(r)
print("RESPONSE:\n{}".format(json.dumps(rv, indent=2)))
open_positions=pd.DataFrame.from_dict(rv['trades'])

#Placing a MarketOrder with TakeProfitOrder and StopLossOrder
import json
from oandapyV20.contrib.requests import MarketOrderRequest
from oandapyV20.contrib.requests import TakeProfitDetails, StopLossDetails
import oandapyV20.endpoints.orders as orders
import oandapyV20

api = oandapyV20.API(access_token=access_token)

mktOrder = MarketOrderRequest(
    instrument="EUR_USD",
    units=999)

# create the OrderCreate request
r = orders.OrderCreate(accountID, data=mktOrder.data)
try:
    # create the OrderCreate request
    rv = api.request(r)
except oandapyV20.exceptions.V20Error as err:
    print(r.status_code, err)
else:
    print(json.dumps(rv, indent=2))
