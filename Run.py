import sys
from TimeSeriesClustering import TimeSeriesKMeansRun, GenerateTimeSeriesDF

if __name__ == '__main__':
    #sys.argv[1] = file_name,sys.argv[2] = date_valiable_name
    #sys.argv[3] = product_valiable_name,sys.argv[4] = values_valiable_nam
    generate_time_series = GenerateTimeSeriesDF(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    time_series_df = generate_time_series.generate_ts_df()

    # sys.argv[5] = n_cluster, sys.argv[6] = metric_name, sys.argv[7] = cluster_valiable_name
    kmeans_run = TimeSeriesKMeansRun(time_series_df, int(sys.argv[5]), sys.argv[6], sys.argv[7])
    cluster_df = kmeans_run.generate_cluster_df()
    
    #sys.argv[8] = upload_file_name
    cluster_df.to_csv(sys.argv[8])