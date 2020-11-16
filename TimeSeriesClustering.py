import pandas as pd
import numpy as np
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesScalerMinMax
# 各商品や各単語などの時系列データ作成クラス
# 行 = 日付, 列 = 各商品名や各単語など, 値 = 数量や売上、検索数など
class GenerateTimeSeriesDF():
    def __init__(self, file_name, date_valiable_name, product_valiable_name, values_valiable_name):
        self.file_name = file_name
        self.date_valiable_name = date_valiable_name
        self.product_valiable_name = product_valiable_name
        self.values_valiable_name = values_valiable_name
    #ファイルを読み込む関数
    def read_file(self):
        if '.csv' in self.file_name:
            dataframe = pd.read_csv(self.file_name)
        elif '.txt' in self.file_name:
            dataframe = pd.read_table(self.file_name, seq=',')
        return dataframe
    #時系列データの作成関数
    def generate_ts_df(self):
        df = self.read_file()
        time_series_df = pd.pivot_table(
          df,
          index=self.date_valiable_name,
          columns=self.product_valiable_name,
          values=self.values_valiable_name,
          aggfunc='sum',
          fill_value=0)
        return time_series_df


#時系列クラスタリングを行う関数
class TimeSeriesKMeansRun():
  def __init__(self, dataframe, n_clusters, metric_name, cluster_name):
    self.dataframe = dataframe
    self.n_clusters = n_clusters
    self.metric_name = metric_name
    self.cluster_name = cluster_name

  #時系列データの配列を整える関数
  def time_series_data_to_array(self):
    tsdata = []
    for i, df in enumerate(self.dataframe):
      tsdata.append(self.dataframe[df].values.tolist()[:])
    tsdata = np.array(tsdata)
    return tsdata

  #データの配列変更関数
  def transform_vector(self):
    tsdata = self.time_series_data_to_array()
    stack_list = []
    for j in range(len(tsdata)):
      data = np.array(tsdata[j])
      data = data.reshape((1, len(data))).T
      stack_list.append(data)
    stack_data = np.stack(stack_list, axis=0)
    return stack_data
  
  #時系列クラスタリングの実行関数
  def time_series_kmeans(self):
    stack_data = self.transform_vector()
    #標準化を行う
    stack_data = TimeSeriesScalerMeanVariance(mu=0.0, std=1.0).fit_transform(stack_data)
    rand_seed = 12345
    # 計算方法がDTWの場合
    if self.metric_name == 'dtw':
      print('run')
      km = TimeSeriesKMeans(n_clusters=self.n_clusters, metric=self.metric_name, n_init=2, random_state=rand_seed)
      dtw_pred = km.fit_predict(stack_data)
      return dtw_pred

    #計算方法がユークリッド距離の場合
    elif self.metric_name == 'euclidean':
      print('run')
      km = TimeSeriesKMeans(n_clusters=self.n_clusters, metric=self.metric_name, random_state=rand_seed)
      euc_pred = km.fit_predict(stack_data)
      return euc_pred
    else:
      print('error')

  #クラスタ番号をデータフレームに結合する関数
  def generate_cluster_df(self):
    dataframe_t = self.dataframe.T
    dataframe_t[self.cluster_name] = self.time_series_kmeans()
    return dataframe_t