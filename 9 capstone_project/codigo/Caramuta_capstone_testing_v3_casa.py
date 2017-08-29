# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 20:12:44 2017

@author: Diego Caramuta
"""

def load_data(path):
    #This function loads the data and splits it into training and test sets
    import pandas as pd    
    global forex, forex_training, forex_testing
    forex = pd.read_table(path, sep=';')
    forex_training = forex[(forex.order <= 92571)] 
    forex_testing = forex[(forex.order > 92571)]
    forex_testing = forex_testing.reset_index()
    forex_testing = forex_testing.drop(['index'], axis=1)
    return 

def balance_insert(timew, USDw, GLw, GLnrw, USDwnet):
    #This function inserts new rows into the balance table
    #The balance table contains the gains and losses of the trading and the balance
    #The Gain/Loss_nreal column contains the gains and losses of the open positions
    #The USD_net column is the USD column plus the Gain/Loss_nreal column
    import pandas as pd
    global balance
    balance_append = pd.DataFrame({'time':[timew],'USD':[USDw],'Gain/Loss_real':[GLw], 'Gain/Loss_nreal':[GLnrw],'USD_net':[USDwnet]})
    balance = balance.append(balance_append, ignore_index=True)
    return

def datetime_insert(timew, datetimew):
    #This function inserts new rows into the datetime_table table
    #This table contains the mapping of the row number and the datetime
    import pandas as pd
    global datetime_table
    datetime_append = pd.DataFrame({'indexx':[timew],'datetime':[datetimew]})
    datetime_table = datetime_table.append(datetime_append, ignore_index=True)
    return

def open_posit_update(new, pos_u, unit_u, price_u):
    #This function inserts new rows into the open positions table or deletes rows when we close open positions
    import pandas as pd
    global open_positions
    if new==1:
        open_positions_append = pd.DataFrame({'position':[pos_u],'units':[unit_u], 'price':[price_u]})
        open_positions = open_positions.append(open_positions_append, ignore_index=True)
    if new==0:
        open_positions = open_positions[open_positions.position != pos_u]
    return

def valid_actions(time, USDreserve, wealth, minutes_history):
    #This function creates the list of valid actions
    #To do nothing is always a valid action
    #If the value of the open positions is not greater than 25 times the net balance then we can buy more positions
    #In addition if the net balance is not less than a certain amount of dollar then we can buy more open positions
    
    #IMPORTANT: this function differs from the one used in the training
    #The difference is that the agent at the beginning is not allowed to trade until it has a sufficient history in order to compute the states
    global actions_list, assets
    actions_list = ['nothing']
    
    assets = 0                 
    for index, row in open_positions[open_positions.position == 'EURUSDbuy'].iterrows():
        assets += EURUSD_bid * row['units']
    for index, row in open_positions[open_positions.position == 'AUDUSDbuy'].iterrows():
        assets += AUDUSD_bid * row['units']
    for index, row in open_positions[open_positions.position == 'GBPUSDbuy'].iterrows():
        assets += GBPUSD_bid * row['units']
    
    if balance.at[time,'USD_net']>USDreserve and assets <= 25 * wealth and len(state_action_hist) >= minutes_history:
        actions_list.append('buyAUDUSD')
        actions_list.append('buyEURUSD')
        actions_list.append('buyGBPUSD')
    return

def howmuchtoinvest(dollarstoinvest, action):
    #If we buy a position we invest a fix ammount of dollars
    #If we sell a position we sell all the positions that we have of this pair currencies
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
    #This function updates the balance table
    #It uses the balance_insert function that we previously described
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

def states(qlearn, minutes_history):
    #This function creates the states
    global state, max_EURUSD, max_GBPUSD, max_AUDUSD, min_EURUSD, min_AUDUSD, min_GBPUSD, EURUSD_hist, AUDUSD_hist, GBPUSD_hist
    state = (0, 0, 0, 0, 0, 0, 0)
    
    if qlearn==1:
    
        #We save some data of historic prices in order to define the states
        EURUSD_append = pd.DataFrame({'EURUSD_ask':[EURUSD_ask]})
        if len(EURUSD_hist)<=minutes_history:
            EURUSD_hist = EURUSD_hist.append(EURUSD_append, ignore_index=True)
        if len(EURUSD_hist)>minutes_history:
            EURUSD_hist = EURUSD_hist.loc[1:]
            EURUSD_hist = EURUSD_hist.append(EURUSD_append, ignore_index=True)
        AUDUSD_append = pd.DataFrame({'AUDUSD_ask':[AUDUSD_ask]})
        if len(AUDUSD_hist)<=minutes_history:
            AUDUSD_hist = AUDUSD_hist.append(AUDUSD_append, ignore_index=True)
        if len(AUDUSD_hist)>minutes_history:
            AUDUSD_hist = AUDUSD_hist.loc[1:]
            AUDUSD_hist = AUDUSD_hist.append(AUDUSD_append, ignore_index=True)
        GBPUSD_append = pd.DataFrame({'GBPUSD_ask':[GBPUSD_ask]})
        if len(GBPUSD_hist)<=minutes_history:
            GBPUSD_hist = GBPUSD_hist.append(GBPUSD_append, ignore_index=True)
        if len(GBPUSD_hist)>minutes_history:
            GBPUSD_hist = GBPUSD_hist.loc[1:]
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
        
        regEUR = sm.ols(formula="EURUSD_ask ~ EURUSD_hist.index.tolist()", data=EURUSD_hist).fit()
        regAUD = sm.ols(formula="AUDUSD_ask ~ AUDUSD_hist.index.tolist()", data=AUDUSD_hist).fit()
        regGBP = sm.ols(formula="GBPUSD_ask ~ GBPUSD_hist.index.tolist()", data=GBPUSD_hist).fit()
        
        #These are the variables that compute the prices'trends:
        if regEUR.params.tolist()[1]<=0:
            rise_EUR = -1
        if regEUR.params.tolist()[1]>0:
            rise_EUR = 1
        if regAUD.params.tolist()[1]<=0:
            rise_AUD = -1
        if regAUD.params.tolist()[1]>0:
            rise_AUD = 1
        if regGBP.params.tolist()[1]<=0:
            rise_GBP = -1
        if regGBP.params.tolist()[1]>0:
            rise_GBP = 1
        
        #This is the variable that indicates which of the trens is the greatest:
        greater_trend = 0
        if regEUR.params.tolist()[1] > regAUD.params.tolist()[1] and regEUR.params.tolist()[1] > regGBP.params.tolist()[1]:
            greater_trend = 3
        if regAUD.params.tolist()[1] > regEUR.params.tolist()[1] and regAUD.params.tolist()[1] > regGBP.params.tolist()[1]:
            greater_trend = 2
        if regGBP.params.tolist()[1] > regEUR.params.tolist()[1] and regGBP.params.tolist()[1] > regAUD.params.tolist()[1]:
            greater_trend = 1
        
        #These are the variables that capture if the prices are near the mean, at the maximum, or at the minimum.
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
        
        state = (rise_EUR, gain_buyEURUSD, rise_AUD, gain_buyAUDUSD, rise_GBP, gain_buyGBPUSD, greater_trend)
    return

def qlearning():
    #This function initializes the Q function
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
    #This function updates the Q function    
    global Q
    Q[state][action] = (1-alpha)*Q[state][action] + alpha*reward
    return


def state_action_history(minutes_history):
    #This function updates the state_Action_history table
    #It inserts new rows and eliminates the oldest one
    #This table contains the history of previous actions
    #The length of the table is fixed
    global state_action_hist    
    state_action_append = pd.DataFrame({'state':[state], 'action':[actual_action], 'price':[tx_price], 'EURUSDa':[EURUSD_ask], 'AUDUSDa':[AUDUSD_ask], 'GBPUSDa':[GBPUSD_ask], 'EURUSDb':[EURUSD_bid], 'AUDUSDb':[AUDUSD_bid], 'GBPUSDb':[GBPUSD_bid]})
    if len(state_action_hist)<=minutes_history:
        state_action_hist = state_action_hist.append(state_action_append, ignore_index=True)
    if len(GBPUSD_hist)>minutes_history:
        state_action_hist = state_action_hist.loc[1:]
        state_action_hist = state_action_hist.append(state_action_append, ignore_index=True)
    return


def update_reward(alpha, qlearn, minutes_history):
    #This function generates the rewards
    #The rewards are computed after some time the action is taken
    #In particular after x minutes have passed then the reward is computed
    #In this sense, we go to the state_action_history table we pick the first action and compute the reward for that action
    global reward, state_reward, action_reward
    if qlearn==1:
        if len(state_action_hist)>=minutes_history:
            action_reward = state_action_hist.at[0, 'action']
            state_reward = state_action_hist.at[0, 'state']
            price_orig = state_action_hist.at[0, 'price']
            EURUSD_orig = state_action_hist.at[0, 'EURUSDa']
            AUDUSD_orig = state_action_hist.at[0, 'AUDUSDa']
            GBPUSD_orig = state_action_hist.at[0, 'GBPUSDa']             
            if action_reward == 'buyEURUSD':
                if len(state_action_hist[state_action_hist.action=='sellEURUSD'])==0:
                    reward=-1
                if len(state_action_hist[state_action_hist.action=='sellEURUSD'])>0:
                    if state_action_hist[state_action_hist.action=='sellEURUSD']['price'].iloc[0] > price_orig:
                        reward=3
                        if ( (state_action_hist[state_action_hist.action=='sellEURUSD']['price'].iloc[0] - price_orig) > (state_action_hist[state_action_hist.action=='sellEURUSD']['AUDUSDb'].iloc[0] - AUDUSD_orig) ) and ( (state_action_hist[state_action_hist.action=='sellEURUSD']['price'].iloc[0] - price_orig) > (state_action_hist[state_action_hist.action=='sellEURUSD']['GBPUSDb'].iloc[0] - GBPUSD_orig) ):
                            reward=5    
                    if state_action_hist[state_action_hist.action=='sellEURUSD']['price'].iloc[0] == price_orig:
                        reward=0
                    if state_action_hist[state_action_hist.action=='sellEURUSD']['price'].iloc[0] < price_orig:
                        reward=-1
                update_q(alpha, state_reward, action_reward, reward)
            if action_reward == 'buyAUDUSD':
                if len(state_action_hist[state_action_hist.action=='sellAUDUSD'])==0:
                    reward=-1
                if len(state_action_hist[state_action_hist.action=='sellAUDUSD'])>0:
                    if state_action_hist[state_action_hist.action=='sellAUDUSD']['price'].iloc[0] > price_orig:
                        reward=3
                        if ( (state_action_hist[state_action_hist.action=='sellAUDUSD']['price'].iloc[0] - price_orig) > (state_action_hist[state_action_hist.action=='sellAUDUSD']['EURUSDb'].iloc[0] - EURUSD_orig) ) and ( (state_action_hist[state_action_hist.action=='sellAUDUSD']['price'].iloc[0] - price_orig) > (state_action_hist[state_action_hist.action=='sellAUDUSD']['GBPUSDb'].iloc[0] - GBPUSD_orig) ):
                            reward=5    
                    if state_action_hist[state_action_hist.action=='sellAUDUSD']['price'].iloc[0] == price_orig:
                        reward=0
                    if state_action_hist[state_action_hist.action=='sellAUDUSD']['price'].iloc[0] < price_orig:
                        reward=-1
                update_q(alpha, state_reward, action_reward, reward)
            if action_reward == 'buyGBPUSD':
                if len(state_action_hist[state_action_hist.action=='sellGBPUSD'])==0:
                    reward=-1
                if len(state_action_hist[state_action_hist.action=='sellGBPUSD'])>0:
                    if state_action_hist[state_action_hist.action=='sellGBPUSD']['price'].iloc[0] > price_orig:
                        reward=3
                        if ( (state_action_hist[state_action_hist.action=='sellGBPUSD']['price'].iloc[0] - price_orig) > (state_action_hist[state_action_hist.action=='sellGBPUSD']['EURUSDb'].iloc[0] - EURUSD_orig) ) and ( (state_action_hist[state_action_hist.action=='sellGBPUSD']['price'].iloc[0] - price_orig) > (state_action_hist[state_action_hist.action=='sellGBPUSD']['AUDUSDb'].iloc[0] - AUDUSD_orig) ):
                            reward=5                            
                    if state_action_hist[state_action_hist.action=='sellGBPUSD']['price'].iloc[0] == price_orig:
                        reward=0
                    if state_action_hist[state_action_hist.action=='sellGBPUSD']['price'].iloc[0] < price_orig:
                        reward=-1
                update_q(alpha, state_reward, action_reward, reward)
    return

def choose_action(learn, min_gain):
    #This function chooses the action that gives the maximum Q given the state
    # or chooses a random action if epsilon is greater than a random number
    # or chooses a random action if the agent is not learning
    # or sells a position if by selling all open positions of a given currency pair gains a benefit of x dollars
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
        

def test(data_path, save_Q_path, load_Q_path, minutes_history, dollarstoinv, USreserve, USDwealth, min_gain, qlearn, alpha):
    #This is the main function for testing
    #It executes the previuous functions
    global sm, random, pd, np, Q, balance, open_positions, datetime_table, max_EURUSD, max_GBPUSD, max_AUDUSD, min_EURUSD, min_AUDUSD, min_GBPUSD, EURUSD_hist, AUDUSD_hist, GBPUSD_hist, state_action_hist, price, units
    
    import pandas as pd
    import numpy as np
    import statsmodels.formula.api as sm
    
    import random
    random.seed(12345)

    load_data(data_path)
    
    ## Next we are going to initialize some of the objects that we will use later
    if qlearn==1:
        Q = np.load(load_Q_path).item() #If we are learning we have to load the Q from the training
    if qlearn==0:
        Q = dict() #If we are not learning we are not going to use the Q
    
    open_positions = pd.DataFrame(columns=['position', 'units', 'price']) #The table that has all open positions
    
    EURUSD_hist = pd.DataFrame(columns=['EURUSD_ask']) #The history of EURUSD prices
    AUDUSD_hist = pd.DataFrame(columns=['AUDUSD_ask']) #The history of AUDUSD prices
    GBPUSD_hist = pd.DataFrame(columns=['GBPUSD_ask']) #The history of GBPUSD prices
    
    state_action_hist = pd.DataFrame(columns=['state', 'action', 'price', 'EURUSDa', 'AUDUSDa', 'GBPUSDa', 'EURUSDb', 'AUDUSDb', 'GBPUSDb']) #The history of previous states, actions and prices 
    
    balance = pd.DataFrame(columns=['time','USD', 'Gain/Loss_real', 'Gain/Loss_nreal', 'USD_net' ]) #How much money have we earned??
    balance_insert(timew = 0, USDw = USDwealth, GLw = 0, GLnrw = 0, USDwnet = USDwealth) ## This function inserts new lines into the balance table
    
    datetime_table = pd.DataFrame(columns=['indexx', 'datetime']) #This table is useful only if you want to keep track of the mapping of each iteration index and the datetime associated to it
    datetime_insert(timew = 0, datetimew = '20170101 0000')  #This function inserts new lines into the previous table
    
    max_EURUSD=0
    max_GBPUSD=0
    max_AUDUSD=0
    min_EURUSD=999999
    min_AUDUSD=999999
    min_GBPUSD=999999
    
    
    ## We finish the initialization
    
    ## Lets start gambling!!!!
    for index, row in forex_testing.iterrows():
        
        global AUDUSD_bid, AUDUSD_ask, EURUSD_bid, EURUSD_ask, GBPUSD_bid, GBPUSD_ask, USDJPY_bid, USDJPY_ask, epsilon, actual_action, actions_list, assets
        
        epsilon = 0.05 ##Exploration factor
        
        # We start reading the prices
        datetime=row['datetime']
        AUDUSD_bid=row['AUDUSD_bid']
        AUDUSD_ask=row['AUDUSD_ask']
        EURUSD_bid=row['EURUSD_bid']
        EURUSD_ask=row['EURUSD_ask']
        GBPUSD_bid=row['GBPUSD_bid']
        GBPUSD_ask=row['GBPUSD_ask']
        USDJPY_bid=row['USDJPY_bid']
        USDJPY_ask=row['USDJPY_ask']
        
        # We define current state       
        states(qlearn, minutes_history)
        
        # If the state does not exist in the qlearning function we add it
        qlearning()
        
        # We insert a new lines into the datetime_table table
        datetime_insert(timew = index+1, datetimew = datetime)
        
        # We identify the valid actions at the current state
        # To do nothing is always an option
        # For example, if the leverage is to high we can not buy new positions
        valid_actions(time=index, USDreserve=USreserve, wealth=balance.at[index,'USD'], minutes_history=minutes_history)
       
        # We update the rewards of previous actions in the Qlearning function 
        # We wait x minutes (minutes_history) in order to see if an action was "good" or "bad"
        update_reward(alpha, qlearn, minutes_history)
        
        # Now we choose an action
        # If by selling a position we gain at least min_gain then we sell it
        # If we dont sell and if we are not learning then a random action is taken
        # If we dont sell and if we are learning then the action that maximize q is taken
        choose_action(learn=qlearn, min_gain=min_gain)
        
        # Now we have to decide how many units to buy or to sell
        howmuchtoinvest(dollarstoinvest=dollarstoinv, action=actual_action)
        
        #We update the balance and update the table of open positions
        #Here is where transaction takes place
        balance_update(index, actual_action, units)
        
        if index%10000 == 0:            
            print "Index:", index, ", % of evolution:", 100*index/112825, ", Assets:", int(assets), ",  Balance is:", int(balance.at[index+1,'USD_net'])
        
        #We keep a history of actions and states
        state_action_history(minutes_history)
    
    print "Index:", index, " FINISH!!!!", ", Assets:", int(assets), ",  Balance is:", int(balance.at[index+1,'USD_net'])        
    
    # When the testing finishes we plot the net balance evolution
    import matplotlib.pyplot as plt
    %matplotlib inline
    plt.plot(balance.time,balance.USD_net)
    plt.ylabel('Balance Testing')
    plt.show()
        
    #Finally, we save the Q if we are learning
    if qlearn==1:
        np.save(save_Q_path, Q)    
    return



test(data_path = 'C:/Users/diego/Desktop/Machine-Learning-Nano/9 capstone_project/codigo/forex2017_top5.txt',
      save_Q_path = 'C:/Users/diego/Desktop/Machine-Learning-Nano/9 capstone_project/codigo/Q_test',
      load_Q_path = 'C:/Users/diego/Desktop/Machine-Learning-Nano/9 capstone_project/codigo/Q.npy',
      minutes_history = 360,
      dollarstoinv = 100, 
      USreserve = 200, 
      USDwealth = 1000, 
      min_gain = 1, 
      qlearn = 1, 
      alpha = 0.5)
