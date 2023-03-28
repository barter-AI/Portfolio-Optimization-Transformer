import numpy as np
import pandas as pd
import glob
import torch
from torch.nn import functional as F
from tqdm import tqdm
from torch.utils.data import Dataset
from sklearn.preprocessing import normalize
import datetime
import gc
eps=1e-8
from pympler.tracker import SummaryTracker
tracker = SummaryTracker()

class PortfolioDataset(Dataset):
    def __init__(self, folder='data/nasdaq100', start_date = '2000-01-01', end_date = '2023-01-01', eval=False):
        super().__init__()
        self.etf_path = glob.glob(f"{folder}/*")
        self.etf_path.sort()
        self.thresh = 10
        self.eval = eval
        self.possible_start_date = np.load("start_date_2015.npy", allow_pickle=True)
        self.start_date = start_date
        self.end_date = end_date

        self.start_date_eval = "2014-12-31"
        self.mid_date_eval = "2015-12-31"
        self.end_date_eval = "2016-12-31"
            
        self.tickers = list(pd.read_csv("data/nasdaq100.csv")["Symbol"])
        # self.valid_stocks = np.load("valid_stocks.npy", allow_pickle=True).tolist()
        self.valid_stocks = []
        self.data = self.check_integrity()  
        self.seq_len = 300
        self.context_len = 300
        self.pred_len = 300
        self.num_stocks = len(self.valid_stocks)
        
        print(f'Created dataset over {self.valid_stocks} stocks, Total count: {len(self.valid_stocks)}')
    
    def process_date(self, a, start_date, end_date):
        try:
            a = a[a.Date >= start_date]
            a = a[a.Date <= end_date]
        except:
            print()
        return a
    
    def check_integrity(self):
        """
        Date align and Sanity check for nan values
        """
        data = pd.read_csv(self.etf_path[0], parse_dates=['Date'])#[['Date', 'Name']]
        stock_name = self.etf_path[0].split("/")[-1].split(".")[0]
        data = data.rename(columns={'Name':f'{data.Name.iloc[0]}'})
        # Process data as per date
        data_full = self.process_date(data, self.start_date, self.end_date)
        data_eval = self.process_date(data, self.start_date_eval, self.end_date_eval)
    
        self.valid_stocks.append(data.columns[-1])
        # Change Closing price to name of stock for clarity
        data_eval = data_eval[[data.columns[0], data.columns[4]]]
        data_eval.rename(columns={data_eval.columns[-1]: f'{stock_name}'}, inplace=True)

        data_full = data_full[[data.columns[0], data.columns[4]]]
        data_full.rename(columns={data_full.columns[-1]: f'{stock_name}'}, inplace=True)

        for i in range(1,len(self.etf_path)):
            stock_name = self.etf_path[i].split("/")[-1].split(".")[0]
            a = pd.read_csv(self.etf_path[i], parse_dates=['Date'])#[['Date', 'Name']]
            a = a.rename(columns={'Name':f'{a.Name.iloc[0]}'})
            temp = self.process_date(a, self.start_date_eval, self.end_date_eval)
            if len(temp) < 100 or temp.Close.isna().sum() > 10:
                continue
            temp = pd.merge(data_eval, temp[["Date", "Close"]], on=['Date'], how='outer', )
            if temp.iloc[:, -1].isna().sum() < 10:
                data_eval = temp
                data_full = pd.merge(data_full, a[["Date", "Close"]], on=['Date'], how='outer', )
                data_full.rename(columns={data_full.columns[-1]: f'{stock_name}'}, inplace=True)
                self.valid_stocks.append(stock_name)
            # if len(self.valid_stocks) >=15:
            #     break
        data_full.sort_values(by=['Date'], inplace=True)
        data_full.reset_index(inplace=True, drop=True)
        return data_full

    def check_columns_integrity(self, a):
        return a.isna().sum()
    
    def get_mask(self, a):
        left_pad = np.zeros(self.seq_len-len(a))
        right_pad = np.ones(len(a))
        nan_mask = ~pd.isnull(pd.DataFrame(a)).iloc[:, 0]
        right_pad = (right_pad*nan_mask)
        attention_mask = np.concatenate([left_pad, right_pad])
        return attention_mask
    
    def pad(self, a, decoder=False):
        if type(a) is not np.ndarray:
            a = a.to_numpy()
        padding = np.zeros((self.seq_len-len(a), a.shape[1]))
        if not decoder: 
            a = np.concatenate([padding, a])
        else: 
            a = np.concatenate([a, padding])
        return a

    def __getitem__(self, idx):
        
        if self.eval:
            start_date = self.start_date_eval
            mid_date = self.mid_date_eval
            end_date = self.end_date_eval
        else:
            # start_date = start_date_list[idx]
            start_date = self.possible_start_date[idx]
            mid_date = start_date + np.timedelta64(1, "Y")
            end_date = start_date + np.timedelta64(2, "Y")

        df = self.process_date(self.data, start_date, end_date)
        df.fillna(0,inplace=True)
        past_values = df[df.Date <= mid_date]
        future_values = df[df.Date > mid_date]

        try:
            past_time = self.pad(np.stack(pd.Series(past_values.Date).apply(lambda x: [x.year, x.month, x.day])))
            future_time = self.pad(np.stack(pd.Series(future_values.Date).apply(lambda x: [x.year, x.month, x.day])), True)
        except:
            print("Br")
        past_values = past_values.iloc[:, 1:]
        # TODO: Future values becomes null so write proper sanity check for this. 
        future_values = future_values.iloc[:, 1:]
        sz = future_values.shape[0]

        idx_pr = past_values.shape[0]-1
        price_list = pd.concat([past_values, future_values]).iloc[idx_pr:idx_pr+246, :].values

        past_values = normalize(past_values)
        future_values = normalize(future_values)
        
        try:
            if future_values[future_values == 0].any():
                print(past_values)
        except:
            pass
        past_values = self.pad(past_values)
        try:
            future_values = self.pad(future_values, True)
        except:
            print()
        
        past_mask = self.get_mask(past_values)
        future_mask = self.get_mask(future_values)       

        # sequence = np.nan_to_num(sequence)
        if len(past_values)>self.seq_len:
            raise ValueError
        # tracker.print_diff()

        gc.collect()
        
        return torch.FloatTensor(past_values), torch.FloatTensor(past_mask), torch.FloatTensor(past_time), torch.FloatTensor(future_values), torch.FloatTensor(future_mask), torch.FloatTensor(future_time), torch.FloatTensor(price_list), torch.tensor([sz])
      
    def __len__(self):
        if self.eval == True:
            return 1
        return len(self.possible_start_date)