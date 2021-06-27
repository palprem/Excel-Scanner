from fastapi import FastAPI, Request, Form
# import uvicorn
# from fastapi.responses import HTMLResponse
# from fastapi.templating import Jinja2Templates
from typing import Optional
import pandas as pd
import mysql.connector as cnx
import datetime as dt
from datetime import timedelta
import numpy as np
# import json
import copy
from bs4 import BeautifulSoup as bs
# import requests
from urllib.request import Request as URLRequest, urlopen
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InputModel(BaseModel):
    input_date: dt.date
    start_time: dt.time
    end_time: dt.time
    option: str
    sector: Optional[str]
    
# directory="templates" here is the folder in which html file is present
# templates = Jinja2Templates(directory="templates")

# @app.get('/')
# def show_form(request: Request):
#     return templates.TemplateResponse('form.html', context={'request': request})
    # result = "Type a number"
    # return templates.TemplateResponse('form1.html', context={'request': request, 'result': result})

@app.post("/")
def form_post(request: Request, input_model: InputModel):
    print('POSTED')
    params_list = scanner_func(input_model.input_date, input_model.start_time, input_model.end_time, input_model.option, input_model.sector)

    if input_model.option=='sector':
        scrips = pd.DataFrame(params_list, columns=['id', 'name', 'sector_pc', 'stock_pc', 'rs_wo_beta', 'stock_oic', 'stock_vspike', 
                                          'high_vs_2days', 'high_reach_2days', 'high_vs_3mo', 'high_reach_3mo', 'high_vs_1yr', 'high_reach_1yr', 
                                          'low_vs_2days', 'low_reach_2days', 'low_vs_3mo', 'low_reach_3mo', 'low_vs_1yr', 'low_reach_1yr'])
    else:
        scrips = pd.DataFrame(params_list, columns=['id', 'name', 'nifty_pc', 'stock_pc', 'rs_wo_beta', 'nifty_oic', 'stock_oic', 'stock_vspike', 
                                          'high_vs_2days', 'high_reach_2days', 'high_vs_3mo', 'high_reach_3mo', 'high_vs_1yr', 'high_reach_1yr', 
                                          'low_vs_2days', 'low_reach_2days', 'low_vs_3mo', 'low_reach_3mo', 'low_vs_1yr', 'low_reach_1yr'])

    json_result = scrips.to_json(orient="records")
    # return templates.TemplateResponse('form1.html', context={'request': request, 'result': result})
    return json_result


# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)


##### FUNC DEFNS #######

def resampler(data, sample_interval, date_col, agg_dict, na_subset):    
    sampled_df = data.resample(sample_interval, on=date_col).aggregate(agg_dict)
    sampled_df.dropna(subset=[na_subset], inplace=True)
    sampled_df.reset_index(inplace=True)
    return sampled_df

def get_time(time_list, time_var, input_date):
    # rev: input_date was added to accepted arguments
    time_range = [((dt.datetime.combine(input_date, time_var))-timedelta(minutes=i)).time() for i in range(-2, 3)]
    intersect = list(set(time_list).intersection(time_range))

    if len(intersect)==1:
        rt_time = intersect[0]
        time_flag = True
    else:
        intersect.sort()
        idx_list = [abs(ele.minute-time_var.minute) for ele in intersect]
        try:
            idx = idx_list.index(min(idx_list))
            rt_time = intersect[idx]
            time_flag = True
        except ValueError as e:
            rt_time = dt.time(0, 0)
            time_flag = False

    return rt_time, time_flag

def calc_metrics(d1, d2, option):
    open = d1['open'].to_list()[0]
    close = d2['close'].to_list()[0]
    pc = round(((close - open)/open)*100, 4)

    if option=='sector':
        oic = np.nan
    else:
        open_oi = d1['oi'].to_list()[0]
        close_oi = d2['oi'].to_list()[0]
        oic = round(((close_oi - open_oi)/open_oi)*100, 4)
        
    return pc, oic

def calc_tnover(data, input_date):
    # rev: input_date was added to accepted arguments
    vdf = data.copy()
    vol_agg_dict = {'instrument_id': 'first', 'volume': 'sum'}
    vol_resample = resampler(vdf, '1D', 'ins_date', vol_agg_dict, 'instrument_id')
    vol_resample['5Davg'] = vol_resample['volume'].rolling(window=5).mean()

    try:
        today = vol_resample[vol_resample['ins_date'].dt.date==input_date]
        tdy_vol = today['volume'].to_list()[0]
        last_5d_vol = vol_resample.iloc[today.index-1]['5Davg'].to_list()[0]
        vc_tdy_5day = round(tdy_vol/last_5d_vol, 3)
    except IndexError as e:
        vc_tdy_5day = np.nan

    return vc_tdy_5day

def scrape_high():
    scrape_url = 'https://www.topstockresearch.com/rt/IndexAnalyser/FUTURESANDOPTIONS/Overview/HighLows/Daily'
    ticker_url = 'F:\excl\stock_tickers.csv'

    req = URLRequest(scrape_url, headers={'User-Agent': 'Mozilla/5.0'})
    webpage = urlopen(req).read()
    soup = bs(webpage)
    table = soup.find(id="results_table").contents

    col = [[child.get_text() for child in direct_children.find_all("td")] for direct_children in list(table)]
    l1 = col.pop(1)
    l2 = [l1[i : i+10] for i in range(0, len(l1), 10)]

    beta_df = pd.DataFrame(l2, columns=['name', 'close', '1wk_low_high', '2wk_low_high', 
                                        '1mo_low_high', '3mo_low_high', '6mo_low_high', 
                                        '1yr_low_high', '2yr_low_high', '5yr_low_high'])
    
    beta_df['3mo_low'] = beta_df['3mo_low_high'].apply(lambda x: float(x.split(' / ')[0]))
    beta_df['3mo_high'] = beta_df['3mo_low_high'].apply(lambda x: float(x.split(' / ')[1]))
    beta_df['1yr_low'] = beta_df['1yr_low_high'].apply(lambda x: float(x.split(' / ')[0]))
    beta_df['1yr_high'] = beta_df['1yr_low_high'].apply(lambda x: float(x.split(' / ')[1]))

    beta_df = beta_df[['name', '3mo_low', '3mo_high', '1yr_low', '1yr_high']]
    beta_df.sort_values(by=['name'], inplace=True)

    stock_tickers = pd.read_csv(ticker_url)
    beta_df['symbol'] = beta_df['name'].map(dict(stock_tickers.values))
    beta_df.reset_index(inplace=True, drop=True)
    return beta_df

def sector_comps(sector):
    sector_map = {'NIFTY AUTO': ['HEROMOTOCO', 'BAJAJ-AUTO', 'TVSMOTOR', 'BALKRISIND', 'EICHERMOT', 'BOSCHLTD', 'TATAMOTORS', 'MARUTI', 'AMARAJABAT', 'EXIDEIND', 'ASHOKLEY', 'M&M', 'MRF', 'MOTHERSUMI', 'BHARATFORG'], 
     'NIFTY BANK': ['AUBANK', 'ICICIBANK', 'KOTAKBANK', 'HDFCBANK', 'IDFCFIRSTB', 'AXISBANK', 'SBIN', 'INDUSINDBK', 'RBLBANK', 'FEDERALBNK', 'PNB', 'BANDHANBNK'], 
     'NIFTY ENERGY': ['ADANIGREEN', 'GAIL', 'HINDPETRO', 'BPCL', 'ONGC', 'TATAPOWER', 'IOC', 'RELIANCE', 'NTPC', 'POWERGRID'], 
     'NIFTY FIN SERVICE': ['CHOLAFIN', 'HDFC', 'ICICIPRULI', 'SRTRANSFIN', 'M & MFIN', 'RECLTD', 'ICICIGI', 'HDFCAMC', 'PFC', 'ICICIBANK', 'MUTHOOTFIN', 'KOTAKBANK', 'HDFCBANK', 'BAJFINANCE', 'BAJAJFINSV', 'AXISBANK', 'SBIN', 'HDFCLIFE', 'SBILIFE', 'PEL'], 
     'NIFTY FMCG': ['MARICO', 'ITC', 'NESTLEIND', 'VBL', 'COLPAL', 'JUBLFOOD', 'PGHH', 'GODREJCP', 'BRITANNIA', 'DABUR', 'HINDUNILVR', 'UBL', 'MCDOWELL-N', 'EMAMILTD', 'TATACONSUM'], 
     'NIFTY IT': ['COFORGE', 'MINDTREE', 'WIPRO', 'MPHASIS', 'INFY', 'TECHM', 'TCS', 'HCLTECH', 'OFSS', 'LTI'], 
     'NIFTY MEDIA': ['DISHTV', 'PVR', 'DBCORP', 'INOXLEISUR', 'TVTODAY', 'TV18BRDCST', 'JAGRAN', 'ZEEL', 'NETWORK18', 'SUNTV'],
     'NIFTY METAL': ['TATASTEEL', 'WELCORP', 'HINDZINC', 'ADANIENT', 'JINDALSTEL', 'MOIL', 'HINDALCO', 'JSWSTEEL', 'COALINDIA', 'NATIONALUM', 'NMDC', 'VEDL', 'SAIL', 'RATNAMANI', 'APLAPOLLO'], 
     'NIFTY PHARMA': ['CIPLA', 'DIVISLAB', 'DRREDDY', 'BIOCON', 'SUNPHARMA', 'TORNTPHARM', 'CADILAHC', 'ALKEM', 'AUROPHARMA', 'LUPIN'], 
     'NIFTY PSU BANK': ['MAHABANK', 'J&KBANK', 'IOB', 'CENTRALBK', 'UCOBANK', 'PSB', 'BANKINDIA', 'INDIANB', 'SBIN', 'CANBK', 'PNB', 'UNIONBANK', 'BANKBARODA'], 
     'NIFTY REALTY': ['PHOENIXLTD', 'HEMIPROP', 'OBEROIRLTY', 'BRIGADE', 'GODREJPROP', 'PRESTIGE', 'IBREALEST', 'SUNTECK', 'DLF', 'SOBHA'], 
     'NIFTY PVT BANK': ['ICICIBANK', 'HDFCBANK', 'KOTAKBANK', 'IDFCFIRSTB', 'AXISBANK', 'INDUSINDBK', 'RBLBANK', 'FEDERALBNK', 'YESBANK', 'BANDHANBNK']}

    return sector_map[sector]

def scanner_func(input_date, start_time, end_time, option, sector=None):

    try:
        stocks_db = cnx.connect(host="164.52.207.158", user="stock", password="stockdata@data", database='stock_production')

        ftr_query = f'select id, ins_date, open, high, low, close, oi from future_all where date(ins_date)="{input_date}";'
        ftr_df = pd.read_sql(ftr_query,stocks_db, parse_dates=['ins_date'])

        vol_query = f'select instrument_id, ins_date, volume from instrument_details where date(ins_date) between "{input_date - dt.timedelta(days=10)}" and "{input_date}";'
        vol_df = pd.read_sql(vol_query,stocks_db, parse_dates=['ins_date'])

        fetch_high_query = f'select id, ins_date, high, low from future_all where date(ins_date) between "{input_date - dt.timedelta(days=5)}" and "{input_date - dt.timedelta(days=1)}";'
        high_df = pd.read_sql(fetch_high_query,stocks_db, parse_dates=['ins_date'])

        sl_query = 'select id, tradingsymbol from instruments where f_n_o=1 and tradingsymbol not like "%NIFTY%";'
        sl_df = pd.read_sql(sl_query,stocks_db)

        if option=='sector':
            sector_query = f'select name, ins_date, open, close from niftysectors where date(ins_date)="{input_date}" and name="{sector}";'
            sector_df = pd.read_sql(sector_query,stocks_db, parse_dates=['ins_date'])
        
        stocks_db.close() 
    except Exception as e:
        stocks_db.close()
        print(str(e))

    temp_st = copy.deepcopy(start_time)
    temp_et = copy.deepcopy(end_time)

    vol_df.drop_duplicates(subset=['instrument_id', 'ins_date'], inplace=True)
    vol_df.reset_index(inplace=True, drop=True)
    vol_df['ins_date'] = vol_df['ins_date'] + timedelta(hours=5, minutes=30)

    if option=='nifty':
        bchmrk_df = ftr_df[ftr_df['id']==417]
    else:
        bchmrk_df = sector_df
        sector_components = sector_comps(sector)
        sl_df = sl_df[sl_df['tradingsymbol'].isin(sector_components)]

    bchmrk_time_list = set(list(bchmrk_df['ins_date'].dt.time))

    if (start_time in bchmrk_time_list) and (end_time in bchmrk_time_list):
        b1 = bchmrk_df[bchmrk_df['ins_date'].dt.time==start_time]
        b2 = bchmrk_df[bchmrk_df['ins_date'].dt.time==end_time]
        bchmrk_pc, bchmrk_oic = calc_metrics(b1, b2, option)
    elif (start_time not in bchmrk_time_list) and (end_time in bchmrk_time_list):
        start_time, time_flag = get_time(bchmrk_time_list, start_time, input_date)
        if time_flag==True:
            b1 = bchmrk_df[bchmrk_df['ins_date'].dt.time==start_time]
            b2 = bchmrk_df[bchmrk_df['ins_date'].dt.time==end_time]
            bchmrk_pc, bchmrk_oic = calc_metrics(b1, b2, option)
        else:
            bchmrk_pc = bchmrk_oic = np.nan
    elif (start_time in bchmrk_time_list) and (end_time not in bchmrk_time_list):
        end_time, time_flag = get_time(bchmrk_time_list, end_time, input_date)
        if time_flag==True:
            b1 = bchmrk_df[bchmrk_df['ins_date'].dt.time==start_time]
            b2 = bchmrk_df[bchmrk_df['ins_date'].dt.time==end_time]
            bchmrk_pc, bchmrk_oic = calc_metrics(b1, b2, option)
        else:
            bchmrk_pc = bchmrk_oic = np.nan
    else:
        start_time, time_flag1 = get_time(bchmrk_time_list, start_time, input_date)
        end_time, time_flag2 = get_time(bchmrk_time_list, end_time, input_date)
        if (time_flag1==True) and (time_flag2==True):
            b1 = bchmrk_df[bchmrk_df['ins_date'].dt.time==start_time]
            b2 = bchmrk_df[bchmrk_df['ins_date'].dt.time==end_time]
            bchmrk_pc, bchmrk_oic = calc_metrics(b1, b2, option)
        else:
            bchmrk_pc = bchmrk_oic = np.nan

    lt_high_df = scrape_high()
    id_dict = dict(sl_df.values)
    params_list = []

    for id, name in id_dict.items():    

        start_time = copy.deepcopy(temp_st)
        end_time = copy.deepcopy(temp_et)
        stock_df = ftr_df[ftr_df['id']==id]
        stock_time_list = set(list(stock_df['ins_date'].dt.time))

        if (start_time in stock_time_list) and (end_time in stock_time_list):
            s1 = stock_df[stock_df['ins_date'].dt.time==start_time]
            s2 = stock_df[stock_df['ins_date'].dt.time==end_time]
            stock_pc, stock_oic = calc_metrics(s1, s2, None)
        elif (start_time not in stock_time_list) and (end_time in stock_time_list):
            start_time, time_flag = get_time(stock_time_list, start_time, input_date)
            if time_flag==True:
                s1 = stock_df[stock_df['ins_date'].dt.time==start_time]
                s2 = stock_df[stock_df['ins_date'].dt.time==end_time]
                stock_pc, stock_oic = calc_metrics(s1, s2, None)
            else:
                stock_pc = stock_oic = np.nan
        elif (start_time in stock_time_list) and (end_time not in stock_time_list):
            end_time, time_flag = get_time(stock_time_list, end_time, input_date)
            if time_flag==True:
                s1 = stock_df[stock_df['ins_date'].dt.time==start_time]
                s2 = stock_df[stock_df['ins_date'].dt.time==end_time]
                stock_pc, stock_oic = calc_metrics(s1, s2, None)
            else:
                stock_pc = stock_oic = np.nan
        else:
            start_time, time_flag1 = get_time(stock_time_list, start_time, input_date)
            end_time, time_flag2 = get_time(stock_time_list, end_time, input_date)
            if (time_flag1==True) and (time_flag2==True):
                s1 = stock_df[stock_df['ins_date'].dt.time==start_time]
                s2 = stock_df[stock_df['ins_date'].dt.time==end_time]
                stock_pc, stock_oic = calc_metrics(s1, s2, None)
            else:
                stock_pc = stock_oic = np.nan

        rs_wo_beta = round(stock_pc - bchmrk_pc, 4)

        stock_high = stock_df[(stock_df['ins_date'].dt.time>=start_time) & (stock_df['ins_date'].dt.time<=end_time)]['high'].max()
        stock_low = stock_df[(stock_df['ins_date'].dt.time>=start_time) & (stock_df['ins_date'].dt.time<=end_time)]['low'].min()

        high_df.drop(high_df[(high_df['ins_date'].dt.time<dt.time(9, 15))].index, inplace = True)
        high_df.reset_index(inplace=True, drop=True)
        high_df.drop(high_df[(high_df['ins_date'].dt.time>dt.time(15, 29))].index, inplace = True)
        high_df.reset_index(inplace=True, drop=True)
        agg_dict = {'id': 'first', 'high': 'max', 'low': 'min'}
        high_resample = resampler(high_df[high_df['id']==id], '1D', 'ins_date', agg_dict, 'id')
        two_days_high = max(high_resample.iloc[-1]['high'], high_resample.iloc[-2]['high'])
        two_days_low = min(high_resample.iloc[-1]['low'], high_resample.iloc[-2]['low'])

        lt_high_id = lt_high_df[lt_high_df['symbol']==name]
        low_3mo = float(lt_high_id['3mo_low'].to_list()[0])
        low_1yr = float(lt_high_id['1yr_low'].to_list()[0])
        high_3mo = float(lt_high_id['3mo_high'].to_list()[0])
        high_1yr = float(lt_high_id['1yr_high'].to_list()[0])

        high_vs_2days = 'True' if stock_high>two_days_high else 'False'
        high_vs_3mo = 'True' if stock_high>high_3mo else 'False'
        high_vs_1yr = 'True' if stock_high>high_1yr else 'False'

        high_reach_2days = round(((two_days_high - stock_high)/stock_high)*100, 4)
        high_reach_3mo = round(((high_3mo - stock_high)/stock_high)*100, 4)
        high_reach_1yr = round(((high_1yr - stock_high)/stock_high)*100, 4)

        low_vs_2days = 'True' if stock_low<two_days_low else 'False'
        low_vs_3mo = 'True' if stock_low<low_3mo else 'False'
        low_vs_1yr = 'True' if stock_low<low_1yr else 'False'
        
        low_reach_2days = round(((two_days_low - stock_low)/stock_low)*100, 4)
        low_reach_3mo = round(((low_3mo - stock_low)/stock_low)*100, 4)
        low_reach_1yr = round(((low_1yr - stock_low)/stock_low)*100, 4)

        stock_vol = vol_df[vol_df['instrument_id']==id].reset_index(drop=True)
        stock_vol.drop(stock_vol[(stock_vol['ins_date'].dt.time<start_time)].index, inplace = True)
        stock_vol.reset_index(inplace=True, drop=True)
        stock_vol.drop(stock_vol[(stock_vol['ins_date'].dt.time>end_time)].index, inplace = True)
        stock_vol.reset_index(inplace=True, drop=True)
        stock_vspike = calc_tnover(stock_vol, input_date)

        if option=='sector':
            params_list.append([id, name, bchmrk_pc, stock_pc, rs_wo_beta, stock_oic, stock_vspike, 
                                high_vs_2days, high_reach_2days, high_vs_3mo, high_reach_3mo, high_vs_1yr, high_reach_1yr, 
                                low_vs_2days, low_reach_2days, low_vs_3mo, low_reach_3mo, low_vs_1yr, low_reach_1yr])
        else:
            params_list.append([id, name, bchmrk_pc, stock_pc, rs_wo_beta, bchmrk_oic, stock_oic, stock_vspike, 
                                high_vs_2days, high_reach_2days, high_vs_3mo, high_reach_3mo, high_vs_1yr, high_reach_1yr, 
                                low_vs_2days, low_reach_2days, low_vs_3mo, low_reach_3mo, low_vs_1yr, low_reach_1yr])
        
        print(id, name)
            
    return params_list

    if _name_ == '_main_':
        uvicorn.run("main:app",host="127.0.0.1", port=8000, reload=True)