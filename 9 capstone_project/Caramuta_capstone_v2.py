# -*- coding: utf-8 -*-
"""
Created on Mon Aug 07 10:55:19 2017

@author: Diego Caramuta
"""

def load_data(path):
    import pandas as pd    
    global forex, forex_toy
    forex = pd.read_table(path, sep=';')
    forex_toy=forex[(forex.order<=100000)]    
    return 

def balance_insert(timew, USDw, GLw, GLnrw, USDwnet):
    import pandas as pd
    global balance
    balance_append = pd.DataFrame({'time':[timew],'USD':[USDw],'Gain/Loss_real':[GLw], 'Gain/Loss_nreal':[GLnrw],'USD_net':[USDwnet]})
    balance = balance.append(balance_append, ignore_index=True)
    return

def datetime_insert(timew, datetimew):
    import pandas as pd
    global datetime_table
    datetime_append = pd.DataFrame({'indexx':[timew],'datetime':[datetimew]})
    datetime_table = datetime_table.append(datetime_append, ignore_index=True)
    return

def open_posit_update(new, pos_u, unit_u, price_u):
    import pandas as pd
    global open_positions
    if new==1:
        open_positions_append = pd.DataFrame({'position':[pos_u],'units':[unit_u], 'price':[price_u]})
        open_positions = open_positions.append(open_positions_append, ignore_index=True)
    if new==0:
        open_positions = open_positions[open_positions.position != pos_u]
    return

def valid_actions(time, USDreserve, wealth):
    global actions_list, assets
    actions_list = ['nothing']
    
    assets = 0                 
    for index, row in open_positions[open_positions.position == 'EURUSDbuy'].iterrows():
        assets += EURUSD_bid * row['units']
    for index, row in open_positions[open_positions.position == 'AUDUSDbuy'].iterrows():
        assets += AUDUSD_bid * row['units']
    for index, row in open_positions[open_positions.position == 'GBPUSDbuy'].iterrows():
        assets += GBPUSD_bid * row['units']
    
    if balance.at[time,'USD_net']>USDreserve and assets <= 25 * wealth:
        actions_list.append('buyAUDUSD')
        actions_list.append('buyEURUSD')
        actions_list.append('buyGBPUSD')
    #if len(open_positions[open_positions.position == 'EURUSDbuy'])>0: 
        #actions_list.append('sellEURUSD')
    #if len(open_positions[open_positions.position == 'GBPUSDbuy'])>0:
        #actions_list.append('sellGBPUSD')
    #if len(open_positions[open_positions.position == 'AUDUSDbuy'])>0:
        #actions_list.append('sellAUDUSD')
    return

def howmuchtoinvest(dollarstoinvest, action):
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
    if action=='sellEURUSD':
        units = sum(open_positions.units[open_positions.position == 'EURUSDbuy'])
    if action=='sellAUDUSD':
        units = sum(open_positions.units[open_positions.position == 'AUDUSDbuy'])
    if action=='sellGBPUSD':
        units = sum(open_positions.units[open_positions.position == 'GBPUSDbuy'])
    return

def balance_update(time, action, units):
    global tx_price
    tx_price=0
    if action=='nothing':
        GL = 0
        newGLnr = 0                 
        for index, row in open_positions[open_positions.position == 'EURUSDbuy'].iterrows():
            newGLnr += (EURUSD_bid - row['price']) * row['units']
        for index, row in open_positions[open_positions.position == 'AUDUSDbuy'].iterrows():
            newGLnr += (AUDUSD_bid - row['price']) * row['units']
        for index, row in open_positions[open_positions.position == 'GBPUSDbuy'].iterrows():
            newGLnr += (GBPUSD_bid - row['price']) * row['units']
        newUSD = balance.at[time,'USD'] + GL
        newGL = balance.at[time,'Gain/Loss_real'] + GL
        newUSDnet = newUSD + newGLnr                     
        balance_insert(timew=time+1, USDw=newUSD, GLw=newGL, GLnrw=newGLnr, USDwnet= newUSDnet)
    if action=='buyEURUSD':
        tx_price=EURUSD_ask
        open_posit_update(new=1, pos_u='EURUSDbuy', unit_u=units, price_u=EURUSD_ask)
        GL = 0
        newGLnr = 0                 
        for index, row in open_positions[open_positions.position == 'EURUSDbuy'].iterrows():
            newGLnr += (EURUSD_bid - row['price']) * row['units']
        for index, row in open_positions[open_positions.position == 'AUDUSDbuy'].iterrows():
            newGLnr += (AUDUSD_bid - row['price']) * row['units']
        for index, row in open_positions[open_positions.position == 'GBPUSDbuy'].iterrows():
            newGLnr += (GBPUSD_bid - row['price']) * row['units']
        newUSD = balance.at[time,'USD'] + GL
        newGL = balance.at[time,'Gain/Loss_real'] + GL
        newUSDnet = newUSD + newGLnr                     
        balance_insert(timew=time+1, USDw=newUSD, GLw=newGL, GLnrw=newGLnr, USDwnet= newUSDnet)
    if action=='sellEURUSD':
        tx_price=EURUSD_bid
        GL = 0
        newGLnr = 0                 
        for index, row in open_positions[open_positions.position == 'EURUSDbuy'].iterrows():
            GL += (EURUSD_bid - row['price']) * row['units']
        for index, row in open_positions[open_positions.position == 'AUDUSDbuy'].iterrows():
            newGLnr += (AUDUSD_bid - row['price']) * row['units']
        for index, row in open_positions[open_positions.position == 'GBPUSDbuy'].iterrows():
            newGLnr += (GBPUSD_bid - row['price']) * row['units']
        newUSD = balance.at[time,'USD'] + GL
        newGL = balance.at[time,'Gain/Loss_real'] + GL
        newUSDnet = newUSD + newGLnr                     
        balance_insert(timew=time+1, USDw=newUSD, GLw=newGL, GLnrw=newGLnr, USDwnet= newUSDnet)
        open_posit_update(new=0, pos_u='EURUSDbuy', unit_u=units, price_u=EURUSD_ask)
    if action=='buyAUDUSD':
        tx_price=AUDUSD_ask
        open_posit_update(new=1, pos_u='AUDUSDbuy', unit_u=units, price_u=AUDUSD_ask)
        GL = 0
        newGLnr = 0                 
        for index, row in open_positions[open_positions.position == 'EURUSDbuy'].iterrows():
            newGLnr += (EURUSD_bid - row['price']) * row['units']
        for index, row in open_positions[open_positions.position == 'AUDUSDbuy'].iterrows():
            newGLnr += (AUDUSD_bid - row['price']) * row['units']
        for index, row in open_positions[open_positions.position == 'GBPUSDbuy'].iterrows():
            newGLnr += (GBPUSD_bid - row['price']) * row['units']
        newUSD = balance.at[time,'USD'] + GL
        newGL = balance.at[time,'Gain/Loss_real'] + GL
        newUSDnet = newUSD + newGLnr                     
        balance_insert(timew=time+1, USDw=newUSD, GLw=newGL, GLnrw=newGLnr, USDwnet= newUSDnet)
    if action=='sellAUDUSD':
        tx_price=AUDUSD_bid
        GL = 0
        newGLnr = 0                 
        for index, row in open_positions[open_positions.position == 'EURUSDbuy'].iterrows():
            newGLnr += (EURUSD_bid - row['price']) * row['units']
        for index, row in open_positions[open_positions.position == 'AUDUSDbuy'].iterrows():
            GL += (AUDUSD_bid - row['price']) * row['units']
        for index, row in open_positions[open_positions.position == 'GBPUSDbuy'].iterrows():
            newGLnr += (GBPUSD_bid - row['price']) * row['units']
        newUSD = balance.at[time,'USD'] + GL
        newGL = balance.at[time,'Gain/Loss_real'] + GL
        newUSDnet = newUSD + newGLnr                     
        balance_insert(timew=time+1, USDw=newUSD, GLw=newGL, GLnrw=newGLnr, USDwnet= newUSDnet)
        open_posit_update(new=0, pos_u='AUDUSDbuy', unit_u=units, price_u=AUDUSD_ask)
    if action=='buyGBPUSD':
        tx_price=GBPUSD_ask
        open_posit_update(new=1, pos_u='GBPUSDbuy', unit_u=units, price_u=GBPUSD_ask)
        GL = 0
        newGLnr = 0                 
        for index, row in open_positions[open_positions.position == 'EURUSDbuy'].iterrows():
            newGLnr += (EURUSD_bid - row['price']) * row['units']
        for index, row in open_positions[open_positions.position == 'AUDUSDbuy'].iterrows():
            newGLnr += (AUDUSD_bid - row['price']) * row['units']
        for index, row in open_positions[open_positions.position == 'GBPUSDbuy'].iterrows():
            newGLnr += (GBPUSD_bid - row['price']) * row['units']
        newUSD = balance.at[time,'USD'] + GL
        newGL = balance.at[time,'Gain/Loss_real'] + GL
        newUSDnet = newUSD + newGLnr                     
        balance_insert(timew=time+1, USDw=newUSD, GLw=newGL, GLnrw=newGLnr, USDwnet= newUSDnet)
    if action=='sellGBPUSD':
        tx_price=GBPUSD_bid
        GL = 0
        newGLnr = 0                 
        for index, row in open_positions[open_positions.position == 'EURUSDbuy'].iterrows():
            newGLnr += (EURUSD_bid - row['price']) * row['units']
        for index, row in open_positions[open_positions.position == 'AUDUSDbuy'].iterrows():
            newGLnr += (AUDUSD_bid - row['price']) * row['units']
        for index, row in open_positions[open_positions.position == 'GBPUSDbuy'].iterrows():
            GL += (GBPUSD_bid - row['price']) * row['units']
        newUSD = balance.at[time,'USD'] + GL
        newGL = balance.at[time,'Gain/Loss_real'] + GL
        newUSDnet = newUSD + newGLnr                     
        balance_insert(timew=time+1, USDw=newUSD, GLw=newGL, GLnrw=newGLnr, USDwnet= newUSDnet)
        open_posit_update(new=0, pos_u='GBPUSDbuy', unit_u=units, price_u=GBPUSD_ask)
    return

#balance_update(time=0, action='sellEURUSD', units=10, EURUSD_bid=1.1, EURUSD_ask=1.2, AUDUSD_bid=1.2, GBPUSD_bid=1.2)    

def states():
    #DEFINE THE STATES
    global state, max_EURUSD, max_GBPUSD, max_AUDUSD, min_EURUSD, min_AUDUSD, min_GBPUSD, EURUSD_hist, AUDUSD_hist, GBPUSD_hist
    state = (0, 0, 0)
    
    #We save 24 hours of historic prices in order to define some variables for the state
    EURUSD_append = pd.DataFrame({'EURUSD_ask':[EURUSD_ask]})
    if len(EURUSD_hist)<=1440:
        EURUSD_hist = EURUSD_hist.append(EURUSD_append, ignore_index=True)
    if len(EURUSD_hist)>1440:
        EURUSD_hist = EURUSD_hist.ix[1:]
        EURUSD_hist = EURUSD_hist.append(EURUSD_append, ignore_index=True)
    AUDUSD_append = pd.DataFrame({'AUDUSD_ask':[AUDUSD_ask]})
    if len(AUDUSD_hist)<=1440:
        AUDUSD_hist = AUDUSD_hist.append(AUDUSD_append, ignore_index=True)
    if len(AUDUSD_hist)>1440:
        AUDUSD_hist = AUDUSD_hist.ix[1:]
        AUDUSD_hist = AUDUSD_hist.append(AUDUSD_append, ignore_index=True)
    GBPUSD_append = pd.DataFrame({'GBPUSD_ask':[GBPUSD_ask]})
    if len(GBPUSD_hist)<=1440:
        GBPUSD_hist = GBPUSD_hist.append(GBPUSD_append, ignore_index=True)
    if len(GBPUSD_hist)>1440:
        GBPUSD_hist = GBPUSD_hist.ix[1:]
        GBPUSD_hist = GBPUSD_hist.append(GBPUSD_append, ignore_index=True)
              
    max_EURUSD=max(max_EURUSD,EURUSD_ask)
    max_GBPUSD=max(max_GBPUSD,GBPUSD_ask)
    max_AUDUSD=max(max_AUDUSD,AUDUSD_ask)
    min_EURUSD=min(min_EURUSD,EURUSD_ask)
    min_GBPUSD=min(min_GBPUSD,GBPUSD_ask)
    min_AUDUSD=min(min_AUDUSD,AUDUSD_ask)
    
    mean_EURUSD=np.mean(EURUSD_hist['EURUSD_ask'])
    std_EURUSD=np.std(EURUSD_hist['EURUSD_ask'])
    mean_AUDUSD=np.mean(AUDUSD_hist['AUDUSD_ask'])
    std_AUDUSD=np.std(AUDUSD_hist['AUDUSD_ask'])
    mean_GBPUSD=np.mean(GBPUSD_hist['GBPUSD_ask'])
    std_GBPUSD=np.std(GBPUSD_hist['GBPUSD_ask'])
    
    gain_buyEURUSD=0
    gain_buyAUDUSD=0
    gain_buyGBPUSD=0
    
    #We generate some variables for the state
    if EURUSD_ask<=mean_EURUSD+std_EURUSD and EURUSD_ask>=mean_EURUSD-std_EURUSD:
        gain_buyEURUSD = 0
    if EURUSD_ask>mean_EURUSD+std_EURUSD:
        gain_buyEURUSD = -1
    if EURUSD_ask<mean_EURUSD-std_EURUSD:
        gain_buyEURUSD = 1
    if EURUSD_ask==min_EURUSD:
        gain_buyEURUSD = 2
    if EURUSD_ask==max_EURUSD:
        gain_buyEURUSD = -2
    if AUDUSD_ask<=mean_AUDUSD+std_AUDUSD and AUDUSD_ask>=mean_AUDUSD-std_AUDUSD:
        gain_buyAUDUSD = 0
    if AUDUSD_ask>mean_AUDUSD+std_AUDUSD:
        gain_buyAUDUSD = -1
    if AUDUSD_ask<mean_AUDUSD-std_AUDUSD:
        gain_buyAUDUSD = 1        
    if AUDUSD_ask==min_AUDUSD:
        gain_buyAUDUSD = 2        
    if AUDUSD_ask==max_AUDUSD:
        gain_buyAUDUSD = -2
    if GBPUSD_ask<=mean_GBPUSD+std_GBPUSD and GBPUSD_ask>=mean_GBPUSD-std_GBPUSD:
        gain_buyGBPUSD = 0
    if GBPUSD_ask>mean_GBPUSD+std_GBPUSD:
        gain_buyGBPUSD = -1
    if GBPUSD_ask<mean_GBPUSD-std_GBPUSD:
        gain_buyGBPUSD = 1
    if GBPUSD_ask==min_GBPUSD:
        gain_buyGBPUSD = 2    
    if GBPUSD_ask==max_GBPUSD:
        gain_buyGBPUSD = -2
    
    state = (gain_buyEURUSD, gain_buyAUDUSD, gain_buyGBPUSD)
    return

def qlearning():
    global Q
    if state in Q:
        pass
    else:
        Q[state] = {}
        Q[state]['nothing'] = 0
        Q[state]['buyEURUSD'] = 0
        Q[state]['buyAUDUSD'] = 0
        Q[state]['buyGBPUSD'] = 0    
    return

def update_q(alpha, state, action, reward):    
    global Q
    Q[state][action] = (1-alpha)*Q[state][action] + alpha*reward
    return

#update_q(0.5, state, actual_action, 1)
#maxQ = max( list (Q[state].values()) )

def state_action_history():
    global state_action_hist    
    state_action_append = pd.DataFrame({'state':[state], 'action':[actual_action], 'price':[tx_price]})
    if len(state_action_hist)<=1440:
        state_action_hist = state_action_hist.append(state_action_append, ignore_index=True)
    if len(GBPUSD_hist)>1440:
        state_action_hist = state_action_hist.ix[1:]
        state_action_hist = state_action_hist.append(state_action_append, ignore_index=True)
    return

def update_reward(alpha):
    global reward, state_reward, action_reward
    if len(state_action_hist)>=1440:
        action_reward = state_action_hist.at[0, 'action']
        state_reward = state_action_hist.at[0, 'state']
        price_orig = state_action_hist.at[0, 'price']
        if action_reward == 'buyEURUSD':
            if len(state_action_hist[state_action_hist.action=='sellEURUSD'])==0:
                reward=-1
            if len(state_action_hist[state_action_hist.action=='sellEURUSD'])>0:
                if state_action_hist[state_action_hist.action=='sellEURUSD']['price'].iloc[0] > price_orig:
                    reward=1
                if state_action_hist[state_action_hist.action=='sellEURUSD']['price'].iloc[0] == price_orig:
                    reward=0
                if state_action_hist[state_action_hist.action=='sellEURUSD']['price'].iloc[0] < price_orig:
                    reward=-2
            update_q(alpha, state_reward, action_reward, reward)
        if action_reward == 'buyAUDUSD':
            if len(state_action_hist[state_action_hist.action=='sellAUDUSD'])==0:
                reward=-1
            if len(state_action_hist[state_action_hist.action=='sellAUDUSD'])>0:
                if state_action_hist[state_action_hist.action=='sellAUDUSD']['price'].iloc[0] > price_orig:
                    reward=1
                if state_action_hist[state_action_hist.action=='sellAUDUSD']['price'].iloc[0] == price_orig:
                    reward=0
                if state_action_hist[state_action_hist.action=='sellAUDUSD']['price'].iloc[0] < price_orig:
                    reward=-2
            update_q(alpha, state_reward, action_reward, reward)
        if action_reward == 'buyGBPUSD':
            if len(state_action_hist[state_action_hist.action=='sellGBPUSD'])==0:
                reward=-1
            if len(state_action_hist[state_action_hist.action=='sellGBPUSD'])>0:
                if state_action_hist[state_action_hist.action=='sellGBPUSD']['price'].iloc[0] > price_orig:
                    reward=1
                if state_action_hist[state_action_hist.action=='sellGBPUSD']['price'].iloc[0] == price_orig:
                    reward=0
                if state_action_hist[state_action_hist.action=='sellGBPUSD']['price'].iloc[0] < price_orig:
                    reward=-2
            update_q(alpha, state_reward, action_reward, reward)
    return

def choose_action(learn, min_gain):
    import random
    global actual_action
    
    if learn==0:
        actual_action=random.choice(actions_list)
    
    if learn==1:      
        r = random.random()
        if r < epsilon:
            actual_action=random.choice(actions_list)
        if r >= epsilon:
            if len(actions_list)>1:
                list_actual_action=[]
                for action in actions_list:
                    if action in ('nothing', 'buyAUDUSD', 'buyEURUSD', 'buyGBPUSD'):
                        if  Q[state][action] == max( list (Q[state].values()) ):
                            list_actual_action.append(action)        
                actual_action=random.choice(list_actual_action)
            if len(actions_list)==1:
                actual_action=random.choice(actions_list)
            
    #If by selling a position we gain at least min_gain then we sell it
    GL_EURUSD=0
    GL_AUDUSD=0
    GL_GBPUSD=0    
    for index, row in open_positions[open_positions.position == 'EURUSDbuy'].iterrows():
        GL_EURUSD += (EURUSD_bid - row['price']) * row['units']
    for index, row in open_positions[open_positions.position == 'AUDUSDbuy'].iterrows():
        GL_AUDUSD += (AUDUSD_bid - row['price']) * row['units']
    for index, row in open_positions[open_positions.position == 'GBPUSDbuy'].iterrows():
        GL_GBPUSD += (GBPUSD_bid - row['price']) * row['units']   
    
    sell_action = []
    if GL_EURUSD>min_gain or GL_AUDUSD>min_gain or GL_GBPUSD>min_gain:
        if GL_EURUSD>min_gain:
            sell_action.append('sellEURUSD')
        if GL_AUDUSD>min_gain:
            sell_action.append('sellAUDUSD')
        if GL_GBPUSD>min_gain:
            sell_action.append('sellGBPUSD')
        actual_action=random.choice(sell_action)
					
    return
        
def main(data_path, dollarstoinv, USreserve, USDwealth, min_gain, qlearn, alpha):
    
    import pandas as pd
    import numpy as np
    
    global pd, np, Q, balance, open_positions, datetime_table, max_EURUSD, max_GBPUSD, max_AUDUSD, min_EURUSD, min_AUDUSD, min_GBPUSD, EURUSD_hist, AUDUSD_hist, GBPUSD_hist, state_action_hist, price, units

    load_data(data_path)
    
    Q = dict() 
    
    open_positions = pd.DataFrame(columns=['position', 'units', 'price'])
    
    EURUSD_hist = pd.DataFrame(columns=['EURUSD_ask'])
    AUDUSD_hist = pd.DataFrame(columns=['AUDUSD_ask'])
    GBPUSD_hist = pd.DataFrame(columns=['GBPUSD_ask'])
    
    state_action_hist = pd.DataFrame(columns=['state', 'action', 'price'])
    
    balance = pd.DataFrame(columns=['time','USD', 'Gain/Loss_real', 'Gain/Loss_nreal', 'USD_net' ])
    balance_insert(timew = 0, USDw = USDwealth, GLw = 0, GLnrw = 0, USDwnet = USDwealth)
    
    datetime_table = pd.DataFrame(columns=['indexx', 'datetime'])
    datetime_insert(timew = 0, datetimew = '20170101 0000')
    
    max_EURUSD=0
    max_GBPUSD=0
    max_AUDUSD=0
    min_EURUSD=999999
    min_AUDUSD=999999
    min_GBPUSD=999999
    for index, row in forex_toy.iterrows():
        
        global AUDUSD_bid, AUDUSD_ask, EURUSD_bid, EURUSD_ask, GBPUSD_bid, GBPUSD_ask, USDJPY_bid, USDJPY_ask, epsilon, actual_action, actions_list, assets
        
        epsilon = 0.9993**(index+1)

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
        ##CUANDO COMPRO, LO HAGO POR X DOLARES
        
        states()
        
        qlearning()
        
        datetime_insert(timew = index+1, datetimew = datetime)
               
        print(index)
        valid_actions(time=index, USDreserve=USreserve, wealth=balance.at[index,'USD'])
        print("               ASSETS: ",int(assets))
        update_reward(alpha)
        choose_action(learn=qlearn, min_gain=min_gain)
        howmuchtoinvest(dollarstoinvest=dollarstoinv, action=actual_action)
        balance_update(index, actual_action, units)
        if index%200 == 0:
            balance.to_csv(r'C:/Users/Diego/Desktop/balance_nolearn.txt', header=True, index=True, sep=' ')
        print("balance is: ", int(balance.at[index+1,'USD_net']))
        print(" ")
        print(" ")
        #falta restarle al balance las comisiones
        state_action_history()
        del AUDUSD_ask, AUDUSD_bid, EURUSD_bid, EURUSD_ask, GBPUSD_bid, GBPUSD_ask, USDJPY_bid, USDJPY_ask
        del actions_list, actual_action, units
    return

import random
random.seed(12345)
main('C:/Users/Diego/Desktop/forex2017_top5.txt',
     dollarstoinv=100, USreserve=200, USDwealth=1000, min_gain=1, qlearn=0, alpha=0.5)

np.save('C:/Users/Diego/Desktop/Q', Q)
Q2 = np.load('C:/Users/Diego/Desktop/Q.npy').item()


table_file = open('C:/Users/Diego/Desktop/Q.txt', 'w') #usar wb si no funciona w
f = table_file
f.write("/-----------------------------------------\n")
f.write("| State-action rewards from Q-Learning\n")
f.write("\-----------------------------------------\n\n")
for state in Q:
    f.write("{}\n".format(state))
    for action, reward in Q[state].items():
        f.write(" -- {} : {:.2f}\n".format(action, reward))
        f.write("\n")
table_file.close()


import matplotlib.pyplot as plt

plt.plot(balance.time,balance.USD)

plt.plot(pd.to_datetime(datetime_table.datetime, format='%Y%m%d %H%M'),balance.USD)


actions_list[1]
wealth.at[1,'USD']
state_action_hist[state_action_hist.action=='sellGBPUSD']
state_action_hist[state_action_hist.action=='sellGBPUSD'].iloc[0]