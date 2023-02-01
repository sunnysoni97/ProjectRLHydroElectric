import pandas as pd
from pandas.core.arrays.period import timedelta
import numpy as np
import os

class Common_Functions():
    
    @staticmethod
    def create_avg_day(df):
        if 'avg_day' not in df.columns:
            df['avg_day']=df.iloc[:, 1:].mean(axis = 1)
        else:
            print("There's already a column avg_day! No need to create another one.")
    
    @staticmethod
    def rename_prices(df):
        if 'PRICES' in df.columns:
            df.rename(columns={"PRICES": "datetime"}, inplace = True)
        else:
            print("There's no column PRICES.")
    
    @staticmethod
    def dataformatting(df):
        Common_Functions.create_avg_day(df)
        Common_Functions.rename_prices(df)
        
        #wide to long
        df = df.melt(id_vars=['datetime'], value_vars=df.columns[1:25]).sort_values(['datetime', 'variable'])
        df.reset_index(inplace=True, drop=True)
        
        #creating master time column, ulgy but works
        time = df['datetime'].copy()
        for d in range(len(df['datetime'])):
            time[d] = df['datetime'][d]+timedelta(hours = d%24) #decided not to go for the +1, so hour 1 is midnight, makes more sense, now it ends in 2009, otherwise the last measurement was 01.01.2010 00:00:00
        df['time'] = time
        
        #hour from string to int
        df['variable'] = df['variable'].map(lambda x:int(x[-2:]))
        
        #renaming, shullfing columns (not important)
        df.rename(columns={"datetime": "date", "variable": "hour", "value":"price"}, inplace = True)

        df['month'] = pd.DatetimeIndex(df['time']).month
        df['day'] = pd.DatetimeIndex(df['time']).day
        df['hour'] = pd.DatetimeIndex(df['time']).hour + 1

        cols = df.columns.tolist()
        cols = [cols[0]] + [cols[3]] + [cols[2]] + [cols[4]] + [cols[5]] + [cols[1]]
        df = df[cols]
        
        return df
    

class Preprocess_Tabular():
    def __init__(self) -> None:
        print("Tabular preprocessor initialised...")
        return

    def discretize_col(self,in_df:pd.DataFrame,col_name:str,n_bins:int,band_length:float) -> pd.DataFrame:
        out_df = in_df.copy()
        max_band = band_length*n_bins
        bins = np.linspace(0,max_band,n_bins)
        bins = np.concatenate([bins,[np.inf]])
        new_col = pd.cut(out_df[col_name],bins,labels=False,include_lowest=True)+1
        out_df[col_name] = new_col
        return out_df

    def create_df_discrete(self,in_df:pd.DataFrame, is_val:bool=False) -> pd.DataFrame:
        out_df = Common_Functions.dataformatting(in_df)
        out_df['month'] = pd.DatetimeIndex(out_df['date']).month
        out_df['price_nondiscrete'] = out_df['price']
        if(is_val):
            out_df = self.discretize_col(out_df,'price',15,11)
        else:
            out_df = self.discretize_col(out_df,'price',15,10)
        return out_df[['price','hour','month','price_nondiscrete']]

    def preprocess_discrete(self, file_path, is_validate:bool=False) -> None:
        
        df = pd.read_excel(file_path)
        base_path = os.path.join(os.path.dirname(__file__),'data')

        if(is_validate):
            base_path = os.path.join(base_path, 'val_data/val_discrete.npy')
        else:
            base_path = os.path.join(base_path, 'train_data/train_discrete.npy')

        out = self.create_df_discrete(df,is_validate)
        
        if(not os.path.exists(os.path.dirname(base_path))):
            os.makedirs(os.path.dirname(base_path))
        
        with open(base_path,'wb') as f:
            np.save(f,out.to_numpy())
        
        print(f"Preprocessed df saved to {base_path}")


class Preprocess_Continous():
    def __init__(self, reduced_form:bool=False) -> None:
        print("Continous preprocessor initialised...")
        pass
    
    def normalize_train(self,df):
        x = df.copy()
        #returns df with all columns normalized individually z-score
        #DANGER - DATA LEAKAGE
        u={}
        for c in range (0, len(x.columns)-1):
            col = x.iloc[:,c].copy()
            norm_col=(col-col.mean())/col.std()
            x.iloc[:,c] = norm_col
            u[str(x.columns[c])]= [col.mean(), col.std()]

        return x, u

    def normalize_validation(self, df, u):
        x = df.copy()
        #returns df with all columns normalized individually z-score
        #DANGER - DATA LEAKAGE

        for c in range (0, len(x.columns)-1):
            col = x.iloc[:,c].copy()
            norm_col=(col-u[str(x.columns[c])][0])/u[str(x.columns[c])][1]
            x.iloc[:,c] = norm_col

        return x

    def bollinger_bands(self, df, hours):
        out = df.copy()
        rolled_mean = out['price'].rolling(hours).mean() 
        rolled_mean.reset_index(inplace=True, drop=True)
        for i in range (1,hours):
            short = out['price'][0:i]
            rolled_mean.loc[i-1] = short.mean()
        
        rolled_std = out['price'].rolling(hours).std() 
        rolled_std.reset_index(inplace=True, drop=True)
        for i in range (1,hours):
            # that's an estimation, first few points are quite bad but evens out quickly
            short = out['price'][0:i+5] #to make it less biased, i+5, so no 0 at the beginning etc
            rolled_std.loc[i-1] = short.std()

        bollinger_up = rolled_mean + rolled_std * 2 # Calculate top band
        bollinger_middle = rolled_mean
        bollinger_down = rolled_mean - rolled_std * 2 # Calculate bottom band

        out[f'{hours}h / {int(hours/24)}day bollinger down'] = bollinger_down
        out[f'{hours}h / {int(hours/24)}day bollinger middle'] = bollinger_middle
        out[f'{hours}h / {int(hours/24)}day bollinger up'] = bollinger_up

        return out

    def get_ema(self, df, period_length:int):
        #returns a new df, which contains new column
        x = df.copy()
        x[f'{period_length}h / {int(period_length/24)}day EMA'] = x['price'].ewm(span=period_length).mean()
        return x

    def get_atr(self, df, period:int):
        x = df.copy()
        maxes = (x.groupby(['date']).max()['price'])
        mins = (x.groupby(['date']).min()['price'])
        closing_prices = df.loc[df['hour'] == 24]['price']

        maxes.reset_index(inplace=True, drop=True)
        mins.reset_index(inplace=True, drop=True)
        closing_prices.reset_index(inplace=True, drop=True)

        yesterday_prices = closing_prices.shift(1)
        yesterday_prices[0] = yesterday_prices[1] #taking care of NA
        
        tr1 = maxes-mins
        tr2 = abs(maxes - yesterday_prices)
        tr3 = abs(mins - yesterday_prices)

        tmp = pd.DataFrame({"A": tr1, "B": tr2, "C":tr3})
        true_range = tmp[["A", "B", "C"]].max(axis=1)
        a_tr = pd.Series([true_range[0]], dtype = np.float64)
        for i in range(1, len(true_range)):
            if i < period:
                a_tr[i] = (a_tr[i-1] * (i-1) + true_range[i]) / i
            else:
                a_tr[i] = (a_tr[i-1] * (period-1) + true_range[i]) / period

        #extending it so it's same for the day
        a_tr = a_tr.loc[a_tr.index.repeat(24)]
        a_tr.reset_index(inplace=True, drop=True)
        x[f'{period*24}h / {int(period)}day ATR'] = a_tr
        return x

    def stochastic_osc(self, df, hours, smoothing):
        out = df.copy()
        maxes = out['price'].rolling(hours).max()
        mins = out['price'].rolling(hours).min()
        sliced = out['price'][0:hours-1]
        mins[0:hours-1] = min(sliced)
        maxes[0:hours-1] = max(sliced)
        oscillator = (out['price'] - mins) / (maxes - mins) * 100
        out[f'{hours}h / {int(hours/24)}stochastic oscillator'] = oscillator
        #usually you use both the oscillator and rolling average of it, this function adds both if bool flag smoothing, hardcoded 8 hours moving avg
        if smoothing:
            osc_smoothed = oscillator.rolling(8).mean()
            #simple NaN handling
            osc_smoothed[0:7] = osc_smoothed[7]
            out[f'{hours}h / {int(hours/24)}smoothed oscillator'] = osc_smoothed
        return out  

    def build_big(self,data_f):
        df = data_f.copy()
        df = Common_Functions.dataformatting(df)
        
        df = self.bollinger_bands(df, 8)
        df = self.bollinger_bands(df, 24)
        df = self.bollinger_bands(df, 120)
        df = self.bollinger_bands(df, 336)
        df = self.bollinger_bands(df, 720)
        
        df = self.get_ema(df, period_length=8)
        df = self.get_ema(df, period_length=24)
        df = self.get_ema(df, period_length=120)
        df = self.get_ema(df, period_length=336)
        df = self.get_ema(df, period_length=720)

        df = self.get_atr(df,5) #here operating on days!
        df = self.get_atr(df,14) #here operating on days!
        df = self.get_atr(df,30) #here operating on days!

        df = self.stochastic_osc(df, 24, True)
        df = self.stochastic_osc(df, 120, True)
        df = self.stochastic_osc(df, 336, True)
        df = self.stochastic_osc(df, 720, True)

        df.drop(columns = ['date', 'time'], inplace = True)

        df['price_unnormalised'] = df['price']

        return df

    def preprocess_big(self, file_path, is_validate:bool=False, columns_to_select:list=[], train_values:dict={},small_path:bool=False) -> dict:
        
        
        df = pd.read_excel(file_path)
        base_path = os.path.join(os.path.dirname(__file__),'data')

        if(is_validate):
            if(small_path):
                base_path = os.path.join(base_path, 'val_data/val_small.npy')
            else:
                base_path = os.path.join(base_path, 'val_data/val_big.npy')
        else:
            if(small_path):
                base_path = os.path.join(base_path, 'train_data/train_small.npy')
            else:
                base_path = os.path.join(base_path, 'train_data/train_big.npy')

        out = self.build_big(df)
        train_values_out = {}

        if(is_validate):
            out = self.normalize_validation(out,u=train_values)
        else:
            out, train_values_out = self.normalize_train(out)


        if(len(columns_to_select)>0):
            out = out[columns_to_select]
        
        if(not os.path.exists(os.path.dirname(base_path))):
            os.makedirs(os.path.dirname(base_path))
        
        with open(base_path,'wb') as f:
            np.save(f,out.to_numpy())
        
        print(f"Preprocessed df saved to {base_path}")
        return train_values_out


    def preprocess_small(self, file_path, is_validate:bool=False, train_values:dict={}) -> tuple:
        columns_to_select = ['price','hour','month','price_unnormalised']
        out = self.preprocess_big(file_path,is_validate,columns_to_select,train_values,small_path=True)
        return out,columns_to_select

