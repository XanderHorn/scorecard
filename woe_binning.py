import pandas as pd
import scipy.stats as stats
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from multiprocessing import Pool
import multiprocessing
from tqdm.contrib.concurrent import process_map

class Binning(BaseEstimator, TransformerMixin):

    def __init__(self, id_col=None, min_bin_pct=0.025, y_threshold=10, p_threshold=0.35, max_pct_missing_threshold=0.8, show_progress=True):

        self.max_pct_missing_threshold = max_pct_missing_threshold
        self.min_bin_pct = min_bin_pct
        self.y_threshold = y_threshold
        self.p_threshold = p_threshold
        self.show_progress = show_progress
        self.id_col = id_col
        self.woe_tables = pd.DataFrame()

        print(f"Using {multiprocessing.cpu_count()} cores")

    def is_numeric_feature(self):
        feature_type = self.data[self.x].dtype
        if feature_type in ['int64', 'int32', 'float64', 'float32']:
            return True
        else:
            return False

    def categorical_binning(self):
    
        tmp_df = pd.DataFrame(self.data[self.x].value_counts())
        tmp_df.reset_index(inplace=True)
        tmp_df.columns = ['feature_value', 'bin_size']
        tmp_df['pct_size'] = tmp_df['bin_size'] / tmp_df['bin_size'].sum()
        tmp_df['feature_value'] = np.where(tmp_df['pct_size'] < self.min_bin_pct, '__ALL_OTHER__', tmp_df['feature_value'])
        tmp_df = tmp_df.groupby('feature_value')['bin_size'].sum().reset_index()
        tmp_df['pct_size'] = tmp_df['bin_size'] / tmp_df['bin_size'].sum()
        return tmp_df['feature_value'].tolist()

    def create_summary(self):
        tmp_df = pd.DataFrame({'x':self.data[self.x], 'y':self.data[self.y]}).groupby('x').agg({'y':['mean','count','std']})
        tmp_df.columns = tmp_df.columns.droplevel(level=0)
        tmp_df.reset_index(inplace=True)
        tmp_df.sort_values('x', ascending=False, inplace=True)

        tmp_df['std'].fillna(0, inplace=True)
        tmp_df.columns = ['feature_value', 'mean_value', 'size', 'std_value']
        return tmp_df
    
    def create_initial_bins(self, data):
        summary = data.copy()
        summary['del_flag'] = 0

        while True:
            i = 0
            summary = summary.loc[summary['del_flag'] != 1]
            summary.reset_index(drop=True, inplace=True)

            while i < len(summary) - 1:
                j = i + 1

                if summary.iloc[j]['mean_value'] >= summary.iloc[i]['mean_value']:
                    while j < len(summary) and summary.loc[j, "mean_value"] >= summary.loc[i, "mean_value"]:

                        n = summary.iloc[j]['size'] + summary.iloc[i]['size']
                        m = (summary.iloc[j]['size'] * summary.iloc[j]['mean_value'] + summary.iloc[i]['size'] * summary.iloc[i]['mean_value']) / n

                        if n == 2:
                            s = np.std([summary.iloc[j]['mean_value'], summary.iloc[i]['mean_value']])
                        else:
                            s = np.sqrt((summary.iloc[j]['size'] * (summary.iloc[j]['std_value'] ** 2) + summary.iloc[i]['size'] * (summary.iloc[i]['std_value'] ** 2)) / n)
                        
                        summary.loc[i, "size"] = n
                        summary.loc[i, "mean_value"] = m
                        summary.loc[i, "std_value"] = s
                        summary.loc[j, "del_flag"] = 1
                        j += 1
                    i = j
                else:
                    i += 1

            if np.sum(summary["del_flag"]) == 0:
                break
        return summary
    
    def combine_bins_based_on_pvalues(self, initial_bins_df):

        bins = initial_bins_df.copy()
        N = bins['size'].sum()
        size_threshold = np.round(N * self.min_bin_pct,0)
        while True:
            bins["mean_value_lead"] = bins["mean_value"].shift(-1)
            bins["size_lead"] = bins["size"].shift(-1)
            bins["std_value_lead"] = bins["std_value"].shift(-1)

            bins["est_size"] = bins["size_lead"] + bins["size"]
            bins["est_mean"] = (bins["mean_value_lead"] * bins["size_lead"] + bins["mean_value"] * bins["size"]) / bins["est_size"]

            bins["est_std_dev2"] = (bins["size_lead"] * bins["std_value_lead"] ** 2 + bins["size"] * bins["std_value"] ** 2) / (bins["est_size"] - 2)

            bins["z_value"] = (bins["mean_value"] - bins["mean_value_lead"]) / np.sqrt(bins["est_std_dev2"] * (1 / bins["size"] + 1 / bins["size_lead"]))

            bins["p_value"] = 1 - stats.norm.cdf(bins["z_value"])

            bins["p_value"] = bins.apply(
                lambda row: row["p_value"] + 1 if (row["size"] < size_threshold) |
                                                  (row["size_lead"] < size_threshold) |
                                                  (row["mean_value"] * row["size"] < self.y_threshold) |
                                                  (row["mean_value_lead"] * row["size_lead"] < self.y_threshold)
                else row["p_value"], axis=1)

            max_p = max(bins["p_value"])
            row_of_maxp = bins['p_value'].idxmax()
            row_delete = row_of_maxp + 1

            if max_p > self.p_threshold:
                bins = bins.drop(bins.index[row_delete])
                bins = bins.reset_index(drop=True)
            else:
                break

            bins["mean_value_lead"] = bins.apply(lambda row: row["est_mean"] if row["p_value"] == max_p else row["mean_value_lead"],
                                             axis=1)
            bins["size"] = bins.apply(
                lambda row: row["est_size"] if row["p_value"] == max_p else row["size"], axis=1)
            bins["std_value"] = bins.apply(
                lambda row: np.sqrt(row["est_std_dev2"]) if row["p_value"] == max_p else row["std_value"], axis=1)

        return bins['feature_value'].tolist()
    
    def bin_array_to_df(self, bin_array):
        if not self.is_numeric_feature():
            bin_array = np.append(bin_array, np.NaN)
            woe_df = pd.DataFrame({
                'feature':self.x,
                'bin':bin_array,
                'from_value_incl':bin_array,
                'to_value_excl':bin_array
            })
            woe_df['bin'] = np.where(woe_df['from_value_incl'].isnull(), 'Missing', woe_df['bin'])
            woe_df['bin'] = np.where(woe_df['bin'] == 'nan', 'Missing', woe_df['bin'])
        else:
            woe_df = pd.DataFrame({
                'feature':self.x,
                'bin':None,
                'from_value_incl':bin_array[1:] + [-np.inf] + [np.NaN],
                'to_value_excl':[np.inf] + bin_array[1:] + [np.NaN]
            })
            woe_df['bin'] = woe_df.apply(lambda row: f'[{row["from_value_incl"]:.2f} : {row["to_value_excl"]:.2f})', axis=1)
            woe_df['bin'] = np.where(woe_df['from_value_incl'].isnull(), 'Missing', woe_df['bin'])
        return woe_df  
    
    def generate_woe_table_from_bin_array(self, bin_array): 
       
        woe_df = self.bin_array_to_df(bin_array)
        tmp_df = self.data.copy()

        tmp_df['bin'] = None
        tmp_df['size'] = 1
        for _, row in woe_df.iterrows():
            from_value = row['from_value_incl']
            to_value = row['to_value_excl']
            bin_value = row['bin']

            if bin_value == 'Missing':
                tmp_df['bin'] = np.where(tmp_df[self.x].isnull(), bin_value, tmp_df['bin'])
            else:
                if self.is_numeric_feature():
                    tmp_df['bin'] = np.where((tmp_df[self.x] >= from_value) & (tmp_df[self.x] < to_value), bin_value, tmp_df['bin'])
                else:
                    tmp_df['bin'] = np.where(tmp_df[self.x] == from_value, bin_value, tmp_df['bin'])

        if not self.is_numeric_feature():
            tmp_df['bin'].fillna('__ALL_OTHER__', inplace = True)

        tmp_df = tmp_df.groupby('bin').agg({
                'size':'sum',
                self.y:'sum',
            }).reset_index()

        woe_df = woe_df.merge(tmp_df, on='bin', how='left')
        woe_df['non_event'] = woe_df['size'] - woe_df[self.y]
        woe_df['pct_size'] = woe_df['size'] / woe_df['size'].sum()

        total_events = woe_df[self.y].sum()
        total_non_events = woe_df['non_event'].sum()

        woe_df['pct_events'] = (woe_df[self.y] + 1) / (total_events + 2)
        woe_df['pct_non_events'] = (woe_df['non_event'] + 1) / (total_non_events + 2)
        woe_df['event_rate'] = woe_df[self.y] / woe_df['size']
        woe_df['woe'] = np.log(woe_df['pct_events'] / woe_df['pct_non_events'])
        woe_df['iv'] = (woe_df['pct_events'] - woe_df['pct_non_events']) * woe_df['woe']
        woe_df['total_iv'] = woe_df['iv'].sum()
        woe_df.rename(columns={self.y:'event'}, inplace=True)
        woe_df['is_numeric_feature'] = self.is_numeric_feature()
        woe_df['woe'].fillna(0 , inplace=True)
        return woe_df

    def bin_feature(self):
        self.is_numeric_feature()

        if not self.is_numeric_feature():
            bins = self.categorical_binning()
        else:
            summary_df = self.create_summary()
            initial_bins = self.create_initial_bins(summary_df)
            bins = self.combine_bins_based_on_pvalues(initial_bins)
        return self.generate_woe_table_from_bin_array(bins)

    def preprocess_df(self):

        features_dtypes = self.data.select_dtypes(include=['int64', 'int32', 'float32', 'float64', 'object', 'bool']).columns
        bools = self.data.select_dtypes(include=['bool']).columns
        if len(bools) > 0:
            for feature in bools:
                self.data[feature] = np.where(self.data[feature] == True, 1, 0)

        summary = pd.DataFrame(index=self.x_cols, columns=['missing_percentage', 'unique_values'])

        for column in self.data.columns:
            missing_percentage = (self.data[column].isnull().sum()) / self.data.shape[0]
            unique_values = self.data[column].nunique()
            
            summary.at[column, 'missing_percentage'] = missing_percentage
            summary.at[column, 'unique_values'] = unique_values
        
        summary = summary.loc[(summary['missing_percentage'] < self.max_pct_missing_threshold) & (summary['unique_values'] > 1)]
        summary.reset_index(inplace=True)
        summary.rename(columns={'index':'feature'}, inplace=True)
        summary = summary.loc[summary['feature'] != self.y]
        try:
            summary = summary.loc[summary['feature'] != self.id_col]
        except:
            pass
        features = summary['feature'].tolist()
        features = [feature for feature in features if feature in features_dtypes]
        return features
    
    def bin_feature_for_pool(self, col):
        if col not in self.id_col:
            self.x = col
            res = self.bin_feature()
            return res
        return None
    
    def bin_data_multicore(self):
        features = self.preprocess_df()

        if self.show_progress:
            results = process_map(self.bin_feature_for_pool, features, max_workers=multiprocessing.cpu_count(), chunksize=multiprocessing.cpu_count())
        else:
            with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
                results = pool.map(self.bin_feature_for_pool, features)

        valid_results = [res for res in results if res is not None]

        self.woe_tables = pd.concat(valid_results, axis=0)

        self.woe_tables['from_value_incl'] = np.where(self.woe_tables['from_value_incl'] == 'nan', None, self.woe_tables['from_value_incl'])
        self.woe_tables['to_value_excl'] = np.where(self.woe_tables['to_value_excl'] == 'nan', None, self.woe_tables['to_value_excl'])

    def bin_data_singlecore(self):

        features = self.preprocess_df()

        for col in features:
            if col not in self.id_col:
                self.x = col
                res = self.bin_feature()
                self.woe_tables = pd.concat([self.woe_tables, res], axis=0)

        self.woe_tables['from_value_incl'] = np.where( self.woe_tables['from_value_incl'] == 'nan', None, self.woe_tables['from_value_incl'])
        self.woe_tables['to_value_excl'] = np.where( self.woe_tables['to_value_excl'] == 'nan', None, self.woe_tables['to_value_excl'])
    
    def fit(self, data, y, id_col=None):
        
        self.data = data.copy()
        self.y = y
        self.id_col = id_col
        self.x_cols = [col for col in data.columns if col not in [self.y, self.id_col]]

        if self.id_col is None:
            self.id_col = []
        self.bin_data_multicore()

    def transform(self, data, woe_tables):
        
        out_df = data.copy()
        for feature in woe_tables['feature'].unique():
            out_df[f"{feature}__WoE__"] = None

            tmp_woe = woe_tables.loc[woe_tables['feature'] == feature].copy()

            for _, row in tmp_woe.iterrows():

                feature = row['feature']
                bin_value = row['bin']
                woe_value = row['woe']
                numeric = row['is_numeric_feature']
                if bin_value == '__ALL_OTHER__':
                    woe_all_other = woe_value
                elif bin_value == 'Missing':
                    woe_missing = woe_value
                
                if numeric and bin_value != 'Missing':
                    from_value = pd.to_numeric(row['from_value_incl'])
                    to_value = pd.to_numeric(row['to_value_excl'])
                else:
                    from_value = row['from_value_incl']
                    to_value = row['to_value_excl']

                if bin_value == 'Missing':
                    continue
                else:
                    if numeric:
                        out_df[f'{feature}__WoE__'] = np.where((out_df[feature] >= from_value) & (out_df[feature] < to_value), woe_value, out_df[f'{feature}__WoE__'])
                    else:
                        out_df[f'{feature}__WoE__'] = np.where(out_df[feature] == from_value, woe_value, out_df[f'{feature}__WoE__'])

            out_df[f'{feature}__WoE__'] = np.where(out_df[feature].isnull(), woe_missing, out_df[f'{feature}__WoE__'])
            try:
                out_df[f'{feature}__WoE__'].fillna(woe_all_other, inplace = True)
            except:
                pass

        cols = [col for col in out_df.columns if '__WoE__' in col]
        if self.id_col:
            cols.append(self.id_col)
        if self.y:
            cols.append(self.y)
            
        out_df = out_df[cols]
        out_df.columns = [col.replace('__WoE__', '') for col in out_df.columns]
        return out_df
