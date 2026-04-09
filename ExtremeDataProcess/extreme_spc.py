import os
import math
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
from io import StringIO
from sklearn.cluster import DBSCAN



def load_spc_data(YEAR, BEGIN_DATETIME, END_DATETIME):
    
    dates = pd.date_range(start=BEGIN_DATETIME, end=END_DATETIME, freq='D')
    dfs = []
    print('start downloading files from Storm Prediction Center')
    for date in tqdm(dates):
        date_str = date.strftime('%Y%m%d')
        for t in ['torn', 'hail', 'wind']:
            url = f"https://www.spc.noaa.gov/climo/reports/{date_str[2:]}_rpts_{t}.csv"
            response = requests.get(url)
            response.raise_for_status() 
            csv_content = response.content.decode('utf-8')
            try:
                df = pd.read_csv(StringIO(csv_content))
                if len(df)>0:
                    df['date'] = date_str
                    df['type'] = t
                    dfs.append(df)
            except Exception as e:
                print(url, e)
    dfs = pd.concat(dfs)
    dfs.columns = [i.lower() for i in dfs.columns] # make column names consistent
    dfs['time'] = dfs['time'].apply(lambda x:str(x).zfill(4))
    dfs['datetime'] = dfs['date'] + dfs['time']
    dfs['datetime'] = pd.to_datetime(dfs['datetime'], format='%Y%m%d%H%M')
    dfs['lon'] = dfs['lon'] + 360
    dfs = dfs.reset_index(drop=True)
    print(dfs.shape)
    print(dfs.columns)
    print(dfs.head(3))
    
    return dfs
    
def make_spc_data(YEAR, BEGIN_DATETIME, END_DATETIME):
    
    # Read data
    df2 = pd.read_csv(f'/data/nihang/weather_data/HRRR/extreme/NOAA-SPC/extreme_data_noaa_spc_{YEAR}_2.csv', parse_dates=['datetime'])
    # print(df2.shape)
    # print(df2.columns)
    # print(df2.head(2))
    # exit(-1)
    
    latlon = np.load('./latlon_grid_hrrr.npy')
    def find_closest_point(grid, new_point):
        distances = np.sqrt(np.sum((grid - new_point) ** 2, axis=2))
        return np.unravel_index(np.argmin(distances), distances.shape)

    def find_closest_points(grid, new_points):
        # lat lon range (21.138123, 52.615654, 225.90453, 299.0828)
        new_points = np.array(new_points)
        new_points = new_points[:, np.newaxis, np.newaxis, :]
        distances = np.sqrt(np.sum((grid - new_points) ** 2, axis=3))
        reshaped_distances = distances.reshape(distances.shape[0], -1)
        min_flat_indices = np.argmin(reshaped_distances, axis=1)
        min_indices = np.unravel_index(min_flat_indices, distances.shape[1:3])
        closest_indices = list(zip(*min_indices))
        return closest_indices


    # Merge events by timestamps, and do spatial clustering to get the extreme event bounding box and remove noisy events
    def fit_dbscan(points, eps=0.2, min_samples=2):    
        points = points.to_numpy()
        mins = np.min(points, axis=0)
        maxs = np.max(points, axis=0)
        standardized_ps = (points-mins) / (maxs-mins+1e-6) 
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        # print(standardized_ps.shape)
        dbscan.fit(standardized_ps)
        return dbscan.labels_ 

    dataset2 = []
    for date in tqdm(df2.datetime.unique()):
        part = df2[df2.datetime==date]
        # print(date)
        if len(part)>= 3: # if few events in a timestamp, remove
            labels = fit_dbscan(part[['BEGIN_LAT','BEGIN_LON']]) # spatial clustering
            if not len(np.unique(labels)) == len(labels): # skip if each point is a cluster
                for label in np.unique(labels):
                    if label!=-1: # label=-1 means it is an outlier, remove it 
                        cluster = part.iloc[labels==label]
                        points = find_closest_points(latlon, cluster[['BEGIN_LAT','BEGIN_LON']].values)
                        points = np.stack(points) # use these points in a cluster to define the bouding box
                        dataset2.append([date,
                                        '+'.join(cluster.type.unique()),
                                        max(points[:,1].min()-20, 0),
                                        max(points[:,0].min()-20, 0),
                                        min(points[:,1].max()+20, 1798),
                                        min(points[:,0].max()+20, 1058)])

    result = pd.DataFrame({
        'begin_time':[i[0] for i in dataset2],
        'end_time':[i[0] for i in dataset2],
        'type':[i[1] for i in dataset2],
        'bounding_box':[str(i[2])+'_'+str(i[3])+'_'+str(i[4])+'_'+str(i[5]) for i in dataset2],
    })

    return result



YEAR = 2024
BEGIN_DATETIME = f"{YEAR}0701"
END_DATETIME = f"{YEAR}1231"

# Load raw SPC data
save_path = f'/data/nihang/weather_data/HRRR/extreme/NOAA-SPC/extreme_data_noaa_spc_{YEAR}_1.csv'
if not os.path.exists(save_path):
    data = load_spc_data(YEAR, BEGIN_DATETIME, END_DATETIME)
    data.to_csv(save_path, index=False)
else:
    data = pd.read_csv(save_path)
exit(-1)
    
# Process SPC data deduplicated by SED data
save_path = f'/data/nihang/weather_data/HRRR/extreme/NOAA-SPC/extreme_data_noaa_spc_{YEAR}.csv'
if not os.path.exists(save_path):
    data = make_spc_data(YEAR, BEGIN_DATETIME, END_DATETIME)
    data.to_csv(save_path, index=False)
else:
    data = pd.read_csv(save_path)