# -*- coding: utf-8 -*-
"""
Created on Mon Aug 07 10:55:19 2017

@author: Diego Caramuta
"""

def load_data(path):
    import pandas as pd    
    global forex, forex_toy
    forex = pd.read_table(path, sep=';')
    forex_toy=forex[(forex.order<=500)]    
    return 

def wealth_update(timew, USDw, EURw, GBPw, AUDw, JPYw, BalanceinUSDw):
    import pandas as pd
    global wealth, wealth_append
    wealth_append = pd.DataFrame({'time':[timew],'USD':[USDw],'EUR':[EURw], 'GBP':[GBPw],'AUD':[AUDw],'JPY':[JPYw],'BalanceinUSD':[BalanceinUSDw]})
    wealth = wealth.append(wealth_append, ignore_index=True)
    return

def valid_actions(time, USDreserve):
    global actions_list
    actions_list = ['nothing']
    if wealth.at[time,'USD']>USDreserve:
        actions_list.append('buyAUDUSD')
        actions_list.append('buyEURUSD')
        actions_list.append('buyGBPUSD')
        actions_list.append('sellUSDJPY')
    if wealth.at[time,'EUR']>0:
        actions_list.append('sellEURUSD')
    if wealth.at[time,'GBP']>0:
        actions_list.append('sellGBPUSD')
    if wealth.at[time,'AUD']>0:
        actions_list.append('sellAUDUSD')
    if wealth.at[time,'JPY']>0:
        actions_list.append('buyUSDJPY')
    return

def choose_action(randomize):
    import random
    global actual_action
    if randomize==1:
        actual_action=random.choice(actions_list)
    if randomize==0:
        if 'buyEURUSD' in actions_list: 
            print 'si'
        else: 
            print 'no'
    return


def howmuchtoinvest(time, dollarstoinvest, action):
    #if we buy a currency different from dollar we invest a fix ammount of dollars
    #if we sell a currency different from dollar we sell the total ammount that we have of this currency
    global units
    units=0
    if action=='buyEURUSD':
        units = int(dollarstoinvest / EURUSD_ask)
    if action=='buyAUDUSD':
        units = int(dollarstoinvest / AUDUSD_ask)
    if action=='buyGBPUSD':
        units = int(dollarstoinvest / GBPUSD_ask)
    if action=='sellUSDJPY':
        units = dollarstoinvest
    if action=='sellEURUSD':
        units = wealth.at[time,'EUR']
    if action=='sellAUDUSD':
        units = wealth.at[time,'AUD']
    if action=='sellGBPUSD':
        units = wealth.at[time,'GBP']
    if action=='buyUSDJPY':
        units = int(wealth.at[time,'JPY'] / USDJPY_ask)
    return

def balance(time, action, units):
    global spreadcost, commission_charge, total_commission
    if action=='buyEURUSD':
        spreadcost = ((EURUSD_ask - EURUSD_bid)/2) * units
        commission_charge = max((units / 100000) * 5 , 0.01)
        total_commission = spreadcost + commission_charge
        newEUR = wealth.at[time,'EUR'] + units
        newUSD = wealth.at[time,'USD'] - (units * EURUSD_ask) - total_commission
        newGBP = wealth.at[time,'GBP']
        newAUD = wealth.at[time,'AUD']
        newJPY = wealth.at[time,'JPY']
    if action=='sellEURUSD':
        spreadcost = ((EURUSD_ask - EURUSD_bid)/2) * units
        commission_charge = max((units / 100000) * 5 , 0.01)
        total_commission = spreadcost + commission_charge
        newEUR = wealth.at[time,'EUR'] - units
        newUSD = wealth.at[time,'USD'] + (units * EURUSD_bid) - total_commission
        newGBP = wealth.at[time,'GBP']
        newAUD = wealth.at[time,'AUD']
        newJPY = wealth.at[time,'JPY']
    if action=='buyAUDUSD':
        spreadcost = ((AUDUSD_ask - AUDUSD_bid)/2) * units
        commission_charge = max((units / 100000) * 5 , 0.01)
        total_commission = spreadcost + commission_charge
        newEUR = wealth.at[time,'EUR'] 
        newUSD = wealth.at[time,'USD'] - (units * AUDUSD_ask) - total_commission
        newGBP = wealth.at[time,'GBP']
        newAUD = wealth.at[time,'AUD'] + units
        newJPY = wealth.at[time,'JPY']
    if action=='sellAUDUSD':
        spreadcost = ((AUDUSD_ask - AUDUSD_bid)/2) * units
        commission_charge = max((units / 100000) * 5 , 0.01)
        total_commission = spreadcost + commission_charge
        newEUR = wealth.at[time,'EUR'] 
        newUSD = wealth.at[time,'USD'] + (units * AUDUSD_bid) - total_commission
        newGBP = wealth.at[time,'GBP']
        newAUD = wealth.at[time,'AUD'] - units
        newJPY = wealth.at[time,'JPY']
    if action=='buyGBPUSD':
        spreadcost = ((GBPUSD_ask - GBPUSD_bid)/2) * units
        commission_charge = max((units / 100000) * 5 , 0.01)
        total_commission = spreadcost + commission_charge
        newEUR = wealth.at[time,'EUR'] 
        newUSD = wealth.at[time,'USD'] - (units * GBPUSD_ask) - total_commission
        newGBP = wealth.at[time,'GBP'] + units
        newAUD = wealth.at[time,'AUD'] 
        newJPY = wealth.at[time,'JPY']
    if action=='sellGBPUSD':
        spreadcost = ((GBPUSD_ask - GBPUSD_bid)/2) * units
        commission_charge = max((units / 100000) * 5 , 0.01)
        total_commission = spreadcost + commission_charge
        newEUR = wealth.at[time,'EUR'] 
        newUSD = wealth.at[time,'USD'] + (units * GBPUSD_bid) - total_commission
        newGBP = wealth.at[time,'GBP'] - units
        newAUD = wealth.at[time,'AUD'] 
        newJPY = wealth.at[time,'JPY']
    if action=='buyUSDJPY':
        spreadcost = ((USDJPY_ask - USDJPY_bid)/2) * units
        commission_charge = max((units / 100000) * 5 , 0.01)
        total_commission = spreadcost + commission_charge
        newEUR = wealth.at[time,'EUR'] 
        newUSD = wealth.at[time,'USD'] + units - total_commission
        newGBP = wealth.at[time,'GBP'] 
        newAUD = wealth.at[time,'AUD'] 
        newJPY = wealth.at[time,'JPY'] - (units * USDJPY_ask)
    if action=='sellUSDJPY':
        spreadcost = ((USDJPY_ask - USDJPY_bid)/2) * units
        commission_charge = max((units / 100000) * 5 , 0.01)
        total_commission = spreadcost + commission_charge
        newEUR = wealth.at[time,'EUR'] 
        newUSD = wealth.at[time,'USD'] - units - total_commission
        newGBP = wealth.at[time,'GBP'] 
        newAUD = wealth.at[time,'AUD'] 
        newJPY = wealth.at[time,'JPY'] + (units * USDJPY_bid)
    if action=='nothing':
        spreadcost = 0
        commission_charge = 0
        total_commission = 0
        newEUR = wealth.at[time,'EUR'] 
        newUSD = wealth.at[time,'USD'] 
        newGBP = wealth.at[time,'GBP'] 
        newAUD = wealth.at[time,'AUD'] 
        newJPY = wealth.at[time,'JPY']
    newBalanceinUSD = newUSD + (EURUSD_bid * newEUR) + (GBPUSD_bid * newGBP) + (AUDUSD_bid * newAUD) + (newJPY/USDJPY_ask)
    wealth_update(timew=time+1, USDw=newUSD, EURw=newEUR, GBPw=newGBP, AUDw=newAUD, JPYw=newJPY, BalanceinUSDw=newBalanceinUSD)
    return

balance(time=1, action='buyEURUSD', units=50, EURUSD_ask=1.1)

def commission(action, units):
    global spreadcost, commission_charge, total_commission
    if action=='buyEURUSD' or action=='sellEURUSD':
        spreadcost = ((EURUSD_ask - EURUSD_bid)/2) * units
    if action=='buyAUDUSD' or action=='sellAUDUSD':
        spreadcost = ((AUDUSD_ask - AUDUSD_bid)/2) * units
    if action=='buyGBPUSD' or action=='sellGBPUSD':
        spreadcost = ((GBPUSD_ask - GBPUSD_bid)/2) * units
    if action=='buyUSDJPY' or action=='sellUSDJPY':
        spreadcost = ((USDJPY_ask - USDJPY_bid)/2) * units
    commission_charge = max((units / 100000) * 5 , 0.01)
    total_commission = spreadcost + commission_charge
    return 


commission(action='buyEURUSD', units=50, AUDUSD_bid=0, AUDUSD_ask=0, EURUSD_bid=1, EURUSD_ask=1.1, GBPUSD_bid=0, GBPUSD_ask=0, USDJPY_bid=0, USDJPY_ask=0)


def main(data_path, dollarstoinv, USreserve, USDwealth, EURwealth, GBPwealth, AUDwealth, JPYwealth):
    
    global wealth

    load_data(data_path)
    
    wealth = pd.DataFrame({'time':0,'USD':[USDwealth],'EUR':[EURwealth], 'GBP':[GBPwealth],'AUD':[AUDwealth],'JPY':[JPYwealth], 'BalanceinUSD':[USDwealth]})
    
    for index, row in forex_toy.iterrows():
        
        global AUDUSD_bid, AUDUSD_ask, EURUSD_bid, EURUSD_ask, GBPUSD_bid, GBPUSD_ask, USDJPY_bid, USDJPY_ask

        datetime=row['datetime']
        AUDUSD_bid=row['AUDUSD_bid']
        AUDUSD_ask=row['AUDUSD_ask']
        EURUSD_bid=row['EURUSD_bid']
        EURUSD_ask=row['EURUSD_ask']
        GBPUSD_bid=row['GBPUSD_bid']
        GBPUSD_ask=row['GBPUSD_ask']
        USDJPY_bid=row['USDJPY_bid']
        USDJPY_ask=row['USDJPY_ask']
        
        ##CUANDO VENDO VENDO LA TOTALIDAD DE LAS DIVISAS DISTINTAS AL DOLAR
        ##CUANDO COMPRO, LO HAGO POR 50 UNIDADES
        
        print index
        valid_actions(time=index, USDreserve=USreserve)
        print actions_list
        choose_action(randomize=1)
        print actual_action
        howmuchtoinvest(time=index, dollarstoinvest=dollarstoinv, action=actual_action)
        print units
        balance(index, actual_action, units)
        print spreadcost, commission_charge, total_commission
        #falta restarle al balance las comisiones
    return


main('U:/Users/dcaramu/Desktop/Machine Learning Nano/9 capstone_project/codigo/forex2017_top5.txt',
     dollarstoinv=50, USreserve=100, USDwealth=1000, EURwealth=0, GBPwealth=0, AUDwealth=0, JPYwealth=0)

actions_list[1]
wealth.at[1,'USD']




