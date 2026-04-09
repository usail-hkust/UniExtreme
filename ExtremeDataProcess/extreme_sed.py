import os
import math
import pandas as pd
import numpy as np
from tqdm import tqdm



def make_sed_data(YEAR, BEGIN_DATETIME, END_DATETIME):
    
    # Load SPC data for repetitive event merging
    old = pd.read_csv(f'/data/nihang/weather_data/HRRR/extreme/NOAA-SPC/extreme_data_noaa_spc_{YEAR}_1.csv', parse_dates=['datetime'])
    old['BEGIN_YEARMONTH'] = old.datetime.apply(lambda x:int(x.strftime("%Y%m")))
    old['BEGIN_DAY'] = old.datetime.apply(lambda x:int(x.strftime("%d")))
    old['location'] = old.location + ', ' + old.state + ', ' + old.county
    old['datetime'] = old['datetime'].dt.floor('H')
    old.drop(columns=['date','time','state','county'],inplace=True)
    df = old[(old['BEGIN_YEARMONTH']>=int(BEGIN_DATETIME[:6]))&(old['BEGIN_YEARMONTH']<=int(END_DATETIME[:6]))]
    df.rename(columns={'lat':'BEGIN_LAT','lon':'BEGIN_LON'},inplace=True)
    
    # Read data and Filter NAN
    # YEAR = 2024
    data = pd.read_csv(f"/data/nihang/weather_data/HRRR/extreme/NOAA-SED/StormEvents_details-ftp_v1.0_d{YEAR}_c20250401.csv")
    data = data[(data.BEGIN_YEARMONTH>=int(BEGIN_DATETIME[:6]))&(data.BEGIN_YEARMONTH<=int(END_DATETIME[:6]))]
    data.BEGIN_LON = data.BEGIN_LON + 360
    data.END_LON = data.END_LON + 360
    chosen_cols = ['EVENT_TYPE','BEGIN_YEARMONTH', 'BEGIN_DAY', 'BEGIN_TIME', 'END_YEARMONTH',
        'END_DAY', 'END_TIME', 'EPISODE_ID', 'EVENT_ID', 'STATE', 
        'BEGIN_RANGE', 'END_RANGE', 'BEGIN_LAT', 'END_LAT', 'BEGIN_LON', 'END_LON',
        'EPISODE_NARRATIVE', 'EVENT_NARRATIVE' ]
    data = data[chosen_cols]
    data.BEGIN_RANGE = data.BEGIN_RANGE*1.61
    data.END_RANGE = data.END_RANGE*1.61
    data = data[~data.BEGIN_RANGE.isna()]
    print('data shape from Storm Event Database:', data.shape)


    # reformulaize datetime
    data['BEGIN_YEARMONTH_str'] = data['BEGIN_YEARMONTH'].astype(str)
    data['BEGIN_DAY_str'] = data['BEGIN_DAY'].astype(str).str.zfill(2)
    data['END_YEARMONTH_str'] = data['END_YEARMONTH'].astype(str)
    data['END_DAY_str'] = data['END_DAY'].astype(str).str.zfill(2)
    data['hour_str'] = data['BEGIN_TIME'].astype(str).str.zfill(4).str[:2]
    data['end_hour_str'] = data['END_TIME'].astype(str).str.zfill(4).str[:2]
    data['begin_date'] = data['BEGIN_YEARMONTH_str'] + data['BEGIN_DAY_str'] + data['hour_str']
    data['end_date'] = data['END_YEARMONTH_str'] + data['END_DAY_str'] + data['end_hour_str']
    data['begin_date'] = pd.to_datetime(data['begin_date'], format='%Y%m%d%H')
    data['end_date'] = pd.to_datetime(data['end_date'], format='%Y%m%d%H')
    data.drop(columns = ['BEGIN_YEARMONTH_str', 'BEGIN_DAY_str', 'hour_str','end_hour_str','END_YEARMONTH_str','END_DAY_str'], inplace=True)
    data.EVENT_TYPE = data.EVENT_TYPE.apply(lambda x:x.replace(' ','_'))
    print('new df shape',data.shape)


    # Merge events into episodes and make bounding boxes
    latlon = np.load('./latlon_grid_hrrr.npy')
    R = 6371.0
    def latitude_difference(distance):
        delta_phi = distance / R
        delta_phi_degrees = math.degrees(delta_phi)
        return delta_phi_degrees

    def longitude_difference(latitude, distance):
        phi = math.radians(latitude)
        delta_lambda = distance / (R * math.cos(phi))
        delta_lambda_degrees = math.degrees(delta_lambda)
        return delta_lambda_degrees

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

    indexes = []
    print('\nstart making bounding boxes')
    bboxes = []
    for ep in tqdm(data.EPISODE_ID.unique()):
        part = data[data.EPISODE_ID==ep]
        points = []
        begin_dates = []
        end_dates = []
        episode_narratives = []
        for idx, row in part.iterrows():
            begin_dates.append(row.begin_date)
            end_dates.append(row.end_date)
            episode_narratives.append(row.EPISODE_NARRATIVE)
            
            # merge repetitive SPC data
            old_part = df[(df.datetime>=row.begin_date)&(df.datetime<=row.end_date)&(df.BEGIN_LAT>row.BEGIN_LAT-1)&
                        (df.BEGIN_LAT<row.BEGIN_LAT+1)&(df.BEGIN_LON>row.BEGIN_LON-1)&(df.BEGIN_LON<row.BEGIN_LON+1)]
            if old_part.shape[0]>0:
                indexes.extend(old_part.index.tolist())
                for _, old_row in old_part.iterrows():
                    points.append([row.BEGIN_LAT, row.BEGIN_LON])
            
            points.append([row.BEGIN_LAT, row.BEGIN_LON]) # center
            if row.BEGIN_RANGE>0:
                diff_lat = latitude_difference(row.BEGIN_RANGE)
                diff_lon = longitude_difference(row.BEGIN_LAT, row.BEGIN_RANGE)
                points.append([row.BEGIN_LAT+diff_lat, row.BEGIN_LON-diff_lon]) # left_upper
                points.append([row.BEGIN_LAT+diff_lat, row.BEGIN_LON+diff_lon]) # right upper
                points.append([row.BEGIN_LAT-diff_lat, row.BEGIN_LON-diff_lon]) # left_lower 
                points.append([row.BEGIN_LAT-diff_lat, row.BEGIN_LON+diff_lon]) # right_lower
            points.append([row.END_LAT, row.END_LON]) # center
            if row.END_RANGE>0:
                diff_lat = latitude_difference(row.END_RANGE)
                diff_lon = longitude_difference(row.END_LAT, row.END_RANGE)
                points.append([row.END_LAT+diff_lat, row.END_LON-diff_lon]) # left_upper
                points.append([row.END_LAT+diff_lat, row.END_LON+diff_lon]) # right upper
                points.append([row.END_LAT-diff_lat, row.END_LON-diff_lon]) # left_lower 
                points.append([row.END_LAT-diff_lat, row.END_LON+diff_lon]) # right_lower
        episode_narratives = set(episode_narratives)
        try:
            assert len(episode_narratives) == 1
        except:
            print(len(episode_narratives))
        episode_narrative = list(episode_narratives)[0]

        types = np.unique(part.EVENT_TYPE.unique())
        types_detail = []
        for t in types:
            type_df = part[part.EVENT_TYPE==t]
            types_detail.append(t + ' ' + type_df.begin_date.min().strftime("%Y%m%d%H") + ' ' + type_df.end_date.max().strftime("%Y%m%d%H"))
        types_detail = ','.join(types_detail)
        types_str = '+'.join(types)
        points = np.array(points)
        points = [[np.min(points[:,0]),np.min(points[:,1])],
                [np.max(points[:,0]),np.max(points[:,1])]] # left_lower & right_upper among all events
        points = np.array(find_closest_points(latlon, points))
        
        # leave some buffer space
        points[0, 0] = max(points[0, 0] - 20, 0) 
        points[0, 1] = max(points[0, 1] - 20, 0) 
        points[1, 0] = min(points[1, 0] + 20, 1058) 
        points[1, 1] = min(points[1, 1] + 20, 1798)
        box_area = (points[1, 0] - points[0, 0]) * (points[1, 1] - points[0, 1])
        region_area = latlon.shape[0] * latlon.shape[1]
        box_region_area_ratio = f"{box_area / region_area * 100:.2f}%"
        bboxes.append([types_str, str(points[0,1])+'_'+str(points[0,0])+'_'+str(points[1,1])+'_'+str(points[1,0]), 
                    box_region_area_ratio, np.min(begin_dates), np.max(end_dates), 
                    (np.max(end_dates) - np.min(begin_dates)).total_seconds() / 3600, 
                    episode_narrative, types_detail]) 
        # break

    merged_data = pd.DataFrame({
        'type': [i[0] for i in bboxes],
        'bounding_box':[i[1] for i in bboxes],
        'area_ratio':[i[2] for i in bboxes],
        'begin_time': [i[3] for i in bboxes],
        'end_time': [i[4] for i in bboxes],
        'duration': [i[5] for i in bboxes],
        'narrative': [i[6] for i in bboxes],
        'details':[i[7] for i in bboxes]
    })
    
    # Remove repetitive events in SPC data for further SPC data processing
    save_path = f'/data/nihang/weather_data/HRRR/extreme/NOAA-SPC/extreme_data_noaa_spc_{YEAR}_2.csv'
    df2 =  df[~df.index.isin(indexes)]
    df2 = df2[df2.BEGIN_YEARMONTH>=int(BEGIN_DATETIME[:6])&(df2.BEGIN_YEARMONTH<=int(END_DATETIME[:6]))]
    df2.to_csv(save_path, index=False)
    
    return merged_data


YEAR = 2024
BEGIN_DATETIME = f"{YEAR}0701"
END_DATETIME = f"{YEAR}1231"
save_path = f'/data/nihang/weather_data/HRRR/extreme/NOAA-SED/extreme_data_noaa_sed_{YEAR}.csv'
MUST_MAKE = True
if not os.path.exists(save_path) or MUST_MAKE:
    data = make_sed_data(YEAR, BEGIN_DATETIME, END_DATETIME)
    data.to_csv(save_path, index=False)
else:
    data = pd.read_csv(save_path)
print(f"\n===\n{data.shape}\n===\n")
    
area_ratio_array = data["area_ratio"].values
area_ratio_array = np.array([eval(e[:-1]) for e in area_ratio_array])
print(area_ratio_array.min(), area_ratio_array.mean(), area_ratio_array.max())
print(data["area_ratio"].value_counts())
print(data["duration"].value_counts())