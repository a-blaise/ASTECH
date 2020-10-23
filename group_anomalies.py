import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from statsmodels.tsa.seasonal import STL
import plotly
from plotly import graph_objs as go
from plotly.subplots import make_subplots
from scipy.spatial import Voronoi
from datetime import datetime
from colour import Color
import math
from sklearn.cluster import DBSCAN
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from statsmodels.tsa.seasonal import seasonal_decompose

class Anomaly:
    def __init__(self, time_slot, port_app, attribute, component, place, coord_x, coord_y, intensity, len_ts):
        self.time_slot = time_slot
        self.port_app = port_app
        self.attribute = attribute
        self.component = component
        self.place = place
        self.coord_x = coord_x
        self.coord_y = coord_y
        self.intensity = intensity
        self.len_ts = len_ts
        
class Spatial_cluster:
    def __init__(self, time_slot, list_cells):
        self.time_slot = time_slot
        self.list_cells = list_cells
        
class Event_signed:
    def __init__(self, start_time, index, list_zones, sign):
        self.start_time = start_time
        self.end_time = start_time
        self.indices = [index]
        self.list_zones = [list_zones]
        self.sign = sign
        
    def update_list_zones(self, new_index, new_list, end_time):
        self.indices.append(new_index)
        self.list_zones.append(new_list)
        self.end_time = end_time
        
def get_X(place):
    if place in fourG_antennas.NOM_SITE.unique():
        return fourG_antennas[fourG_antennas.NOM_SITE == place].COORD_X.values[0] 
    elif place in threeG_antennas.NOM_SITE.unique():
        return threeG_antennas[threeG_antennas.NOM_SITE == place].COORD_X.values[0] 
    elif place in twoG_antennas.NOM_SITE.unique():
        return twoG_antennas[twoG_antennas.NOM_SITE == place].COORD_X.values[0]
    else: return np.nan

def get_Y(place):
    if place in fourG_antennas.NOM_SITE.unique():
        return fourG_antennas[fourG_antennas.NOM_SITE == place].COORD_Y.values[0] 
    elif place in threeG_antennas.NOM_SITE.unique():
        return threeG_antennas[threeG_antennas.NOM_SITE == place].COORD_Y.values[0] 
    elif place in twoG_antennas.NOM_SITE.unique():
        return twoG_antennas[twoG_antennas.NOM_SITE == place].COORD_Y.values[0]
    else: return np.nan

def thresholding_algo(orig, port_app, att, site, COORD_X, COORD_Ys):
    # time series decomposition using STL with parameter robust=True to reduce the impact of outliers
    y = STL(orig, period=FREQ, robust=True).fit().resid
    avgFilter = np.zeros(len(y))
    stdFilter = np.zeros(len(y))
    avgFilter[LAG - 1] = np.mean([y[ind] for ind in range(LAG) if orig[ind] != 0])
    stdFilter[LAG - 1] = np.std([y[ind] for ind in range(LAG) if orig[ind] != 0])
    
    list_anos = []
    for i in range(LAG, len(y) - 1):
        if orig[i] != 0:
            new_ts = [y[ind] for ind in range(i-LAG-1, i-1) if orig[ind] != 0]
            avgFilter[i-1] = np.mean(new_ts)
            stdFilter[i-1] = np.std(new_ts)
            if abs(y[i] - avgFilter[i-1]) > THRESHOLD * stdFilter[i-1]:
                list_anos.append(Anomaly(orig.index[i], port_app, att, site, COORD_X, COORD_Y,
                                         (y[i] - avgFilter[i-1]) / stdFilter[i-1], len(new_ts)))
    return list_anos

def get_period(x):
    rep = 'weekend' if x['day_of_week'] == 'Saturday' or x['day_of_week'] == 'Sunday' else 'weekday'
    rep += ' day' if 7 <= x.hour_of_day < 19 else ' night'
    return rep

def get_groups_signed(df_total, placeList):
    in_group_T_C_iqr_pos, in_group_T_C_iqr_neg = (dict() for i in range(2))
    for key in sorted(placeList.keys()):
        subset_data = df_total[df_total.place == key]
        snapshots = subset_data.groupby(['time_slot']).agg({'intensity': [count_pos, count_neg]}).reset_index()
        snapshots.columns = ['_'.join(col).strip() for col in snapshots.columns.values]
        
        T_c_IQR_pos = np.percentile(snapshots.intensity_count_pos.values, 75, interpolation = 'midpoint') + 1.5 * (np.percentile(snapshots.intensity_count_pos.values, 75, interpolation = 'midpoint') - np.percentile(snapshots.intensity_count_pos.values, 25, interpolation = 'midpoint'))
        T_c_IQR_neg = np.percentile(snapshots.intensity_count_neg.values, 75, interpolation = 'midpoint') + 1.5 * (np.percentile(snapshots.intensity_count_neg.values, 75, interpolation = 'midpoint') - np.percentile(snapshots.intensity_count_neg.values, 25, interpolation = 'midpoint'))
        
        dates = [snapshots.time_slot_.values[0], snapshots.time_slot_.values[-1]]
        period = pd.date_range(*dates, freq='30min')     
        snapshots = snapshots.set_index('time_slot_')
        snapshots.index = pd.DatetimeIndex(snapshots.index)
        snapshots = snapshots.reindex(period, fill_value=0)
        snapshots = snapshots.reset_index()
        snapshots = snapshots.rename(columns={'index': 'time_slot_'})

        if T_c_IQR_pos != 0:
            snapshots_sel_pos = snapshots[snapshots.intensity_count_pos >= T_c_IQR_pos]
            in_group_T_C_iqr_pos[key] = snapshots_sel_pos.time_slot_.values
            
        if T_c_IQR_neg != 0:
            snapshots_sel_neg = snapshots[snapshots.intensity_count_neg >= T_c_IQR_neg]
            in_group_T_C_iqr_neg[key] = snapshots_sel_neg.time_slot_.values
        
    return in_group_T_C_iqr_pos, in_group_T_C_iqr_neg

def rec_get_neighbors(abnormal_snapshot, neighbors_tostudy, studied_neighbors, in_group_T_C_iqr, placeList):
    new_neighbors_to_study = set()
    unknown_neighbors = [n for n in neighbors_tostudy if n not in studied_neighbors]
    if unknown_neighbors:
        for neighbor in unknown_neighbors:
            if neighbor in in_group_T_C_iqr:
                if abnormal_snapshot in in_group_T_C_iqr[neighbor]:
                    studied_neighbors.add(neighbor)
                    new_neighbors_to_study.update([nn for nn in placeList[neighbor] if nn not in studied_neighbors])
        return rec_get_neighbors(abnormal_snapshot, new_neighbors_to_study, studied_neighbors, in_group_T_C_iqr, placeList)
    return studied_neighbors

def get_prop_clusters(placeList, in_group_T_C_iqr):
    list_prop_clusters = []
    for place, neighbors in placeList.items():
        if place in in_group_T_C_iqr:
            for abnormal_snapshot in in_group_T_C_iqr[place]:
                all_relative_neighbors = rec_get_neighbors(abnormal_snapshot, neighbors, {place}, in_group_T_C_iqr, placeList)
                for cluster in list_prop_clusters:
                    if cluster.time_slot == abnormal_snapshot and place in cluster.list_cells:
                        break
                else:
                    list_prop_clusters.append(Spatial_cluster(abnormal_snapshot, all_relative_neighbors))
                
    size_spatial_clusters = [len(cluster.list_cells) for cluster in list_prop_clusters]
    return list_prop_clusters

def voronoi_finite_polygons_2d(vor, radius=None):
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            new_regions.append(vertices)
            continue

        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue

            t = vor.points[p2] - vor.points[p1]
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)

def count_pos(x):
    return len([y for y in x if y>0])

def count_neg(x):
    return len([y for y in x if y<0])

class Label:
    def __init__(self, start_label, end_label, type_label, tag_label):
        self.start = datetime.strptime(start_label, '%Y-%m-%d %H:%M:%S')
        self.end = datetime.strptime(end_label, '%Y-%m-%d %H:%M:%S')
        self.type = type_label
        self.tag = tag_label
        
def label_event(ts):
    event = 'unknown'
    ts_st = ts['start_time']
    ts_end = ts['end_time']
    for type_ev, events in highlights.items():
        for ev in events:
            if ev.start <= ts_st <= ev.end:
                event = ev.type
                break
            elif ev.start <= ts_end <= ev.end:
                event = ev.type
                break
    return event

def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    if len(s1.union(s2)) != 0:
        return len(s1.intersection(s2)) / len(s1.union(s2))
    else:
        return 0
    
def get_list_applis_all(x, dataframe):
    list_total = []
    for ind_ts, zones in enumerate(x['list_zones']):
        ts = x['start_time'] + np.timedelta64(30 * ind_ts, 'm')
        for zone in zones:
            list_total.extend([mapping.loc[mapping.id_port == port]['Application'].values[0] for port in dataframe[(dataframe.place == zone) & (dataframe.time_slot == ts)]['port_app'].values])
    return list_total

def get_voronoi_diagram(df_SDF):
    points = {place: [get_X(place), get_Y(place)] for place in dataframe.place.unique()}
    list_places = list(points.keys())
    coord_points = list(points.values())
    vor = Voronoi(coord_points)
    regions, vertices = voronoi_finite_polygons_2d(vor)
    name_region = {list_places[ind_region]: region for ind_region, region in enumerate(regions)}

    df_SDF['traffic'] = df_SDF['s_nPacketUp'] + df_SDF['s_nPacketDn']
    nb_flows = df_SDF.groupby(['NOM_SITE'])['traffic'].sum()

    nb_flows = nb_flows.sort_values(ascending=False).to_frame('nb_flows').reset_index()

    colors = list(Color('purple').range_to(Color('yellow'),len(nb_flows.values)))
    nb_flows['color'] = colors

    fig, ax = plt.subplots(figsize=[6, 6], dpi=400)
    ax.imshow(plt.imread('map.png'), extent=[vor.min_bound[0] - 0.1, vor.max_bound[0] + 0.1, vor.min_bound[1] - 0.1, vor.max_bound[1] + 0.1])
    for place, region in name_region.items():
        polygon = vertices[region]
        col = str(nb_flows[nb_flows.NOM_SITE==place]['color'].values[0]).ljust(7, '0') if str(nb_flows[nb_flows.NOM_SITE==place]['color'].values[0])[0] == '#' else str(nb_flows[nb_flows.NOM_SITE==place]['color'].values[0])
        plt.fill(*zip(*polygon), facecolor=col, edgecolor='black', alpha=0.2)

    for i, p in enumerate(coord_points):
        if i == 0:
            plt.plot(p[0], p[1], 'ko', ms= 3, color='blue', label='position of the antenna')
        else:
            plt.plot(p[0], p[1], 'ko', ms= 3, color='blue')
    plt.plot(coord_points[0], coord_points[1], 'ko', label='Voronoi cell')
    plt.xlim(vor.min_bound[0] - 0.1, vor.max_bound[0] + 0.1)
    plt.ylim(vor.min_bound[1] - 0.1, vor.max_bound[1] + 0.1)
    plt.xlabel('x coordinate', fontsize=10)
    plt.ylabel('y coordinate', fontsize=10)
    plt.legend(bbox_to_anchor=(0,1.0,1,0.01), loc='lower left', fontsize=10, framealpha=1, ncol=2)
    fig.tight_layout()
    fig.savefig('voronoi_map.png')
    plt.close()

def example_decomp_ts(df_SDF):
    FONT_SIZE = 14
    att = 's_nPacketDn'
    port_app = 65805
    decompfreq = int(24*60/30*7)
    dates = ['2019-03-16', '2019-06-08']
    week_labels = ['2019/03/16', '2019/03/23', '2019/03/30', '2019/04/06', '2019/04/13', '2019/04/20',
                  '2019/04/27', '2019/05/04', '2019/05/11', '2019/05/18', '2019/05/25', '2019/06/01', '2019/06/08']

    df_SDF = df_SDF.sort_values('TimeSlot', axis=0)
    temp = df_SDF[df_SDF.PortApp == port_app]
    temp.drop_duplicates(subset='TimeSlot', inplace=True)
    temp = temp.set_index('TimeSlot')
    temp.index = pd.DatetimeIndex(temp.index)
    temp = temp.reindex(pd.date_range(*dates, freq='30min'), fill_value=0)
    temp = temp.loc[dates[0]:dates[1]]

    list_colors = ['#edf2fb', '#e2eafc', '#d7e3fc', '#ccdbfd']

    for method in ['MA', 'STL', 'STL_robust']:
        if method == 'MA':
            result = seasonal_decompose(temp[att].values, period=decompfreq, model='additive', two_sided=False)
        elif method == 'STL':
            result = STL(temp[att].values, period=decompfreq).fit()
        elif method == 'STL_robust':
            result = STL(temp[att].values, period=decompfreq, robust=True).fit()
        
        fig, ax = plt.subplots(4, figsize=(9, 9), dpi=400, gridspec_kw={'wspace':0, 'hspace':0})
        ax[0].plot(result.observed[2000:], c='black', label='Observed')
        ax[0].set_facecolor(list_colors[0])
        ax[1].plot(result.trend[2000:], c='black', label='Trend')
        ax[1].set_facecolor(list_colors[1])
        ax[2].plot(result.seasonal[2000:], c='black', label='Seasonal')
        ax[2].set_facecolor(list_colors[2])
        ax[3].plot(result.resid[2000:], c='black', label='Residual')
        ax[3].set_facecolor(list_colors[3])
        
        ax[3].set_xticks(range(0, len(result.observed[2000:]), decompfreq))
        ax[3].set_xticklabels([wk[5:] for wk in week_labels[-7:]], fontsize=FONT_SIZE)
        ax[3].xaxis.set_tick_params(labelsize=FONT_SIZE)
        
        for axx in ax:
            axx.legend(fontsize=FONT_SIZE+4, loc='upper left')
            axx.yaxis.set_tick_params(labelsize=FONT_SIZE)
            axx.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            
        fig.tight_layout()
        fig.savefig(method + '.png')
        plt.close()

def plot_map_anomalies(df_anomaly, spatial_clusters_pos, spatial_clusters_neg):
    # use then command "ffmpeg -r 10 -i img_%05d.png -c:v libx264 -vf fps=25 -crf 17 -pix_fmt yuv420p output.mp4" to pour concatenate figures into a video
    points = {place: [get_X(place), get_Y(place)] for place in df_anomaly.place.unique()}
    list_places = list(points.keys())

    coord_points = list(points.values())
    vor = Voronoi(coord_points)
    regions, vertices = voronoi_finite_polygons_2d(vor)
    name_region = {list_places[ind_region]: region for ind_region, region in enumerate(regions)}

    list_ts = list(spatial_clusters_pos.time_slot.unique()) + list(spatial_clusters_neg.time_slot.unique())
    list_ts = list(set(list_ts))
    list_ts.sort()
    os.mkdir('vid_signed')
    for ind_ts, ts in enumerate(list_ts):
        fig, ax = plt.subplots(figsize=[10, 10], dpi=200)
        ax.imshow(plt.imread('map.png'), extent=[vor.min_bound[0] - 0.1, vor.max_bound[0] + 0.1, vor.min_bound[1] - 0.1, vor.max_bound[1] + 0.1])
        for region in regions:
            polygon = vertices[region]
            plt.fill(*zip(*polygon), facecolor='#FFFFFF', edgecolor='black', alpha=0.4)
        for index, row in spatial_clusters_pos[spatial_clusters_pos.time_slot == ts].iterrows():
            for region in regions:
                if region in [name_region[place] for place in row.list_cells if place != 'EDF_PLEYEL_INDOOR']:
                    polygon = vertices[region]
                    plt.fill(*zip(*polygon), facecolor='b', edgecolor='black', alpha=0.4)
        for index, row in spatial_clusters_neg[spatial_clusters_neg.time_slot == ts].iterrows():
            for region in regions:
                if region in [name_region[place] for place in row.list_cells if place != 'EDF_PLEYEL_INDOOR']:
                    polygon = vertices[region]
                    plt.fill(*zip(*polygon), facecolor='y', edgecolor='black', alpha=0.4)

        for i, p in enumerate(coord_points):
            plt.plot(p[0], p[1], 'ko', ms= 1.5, color='blue')
        plt.xlim(vor.min_bound[0] - 0.1, vor.max_bound[0] + 0.1)
        plt.ylim(vor.min_bound[1] - 0.1, vor.max_bound[1] + 0.1)
        plt.title(ts)
        plt.savefig('vid_signed/img_' + str(ind_ts) + '.png')
        plt.close()

def spatiotemporal_events_visualization(st_clusters_pos, st_clusters_neg):
    FONT_SIZE = 12
    colors = ['grey', '#e377c2', 'blue', 'orange']

    fig, ax = plt.subplots(figsize=[6.7, 4.1], dpi=400)
    for ind_lab, lab in enumerate(['unknown', 'match', 'national event', 'appli update']):
        tt = st_clusters_pos[st_clusters_pos.label==lab]
        sizes = tt.groupby(['mean_el', 'len_thread']).count()['sign'].reset_index()
        tt['sizee'] = tt.apply(lambda x: sizes[(sizes.mean_el==x['mean_el']) & (sizes.len_thread==x['len_thread'])].sign.values[0], axis=1)
        size = [15 + 100 * math.log(x) for x in tt.sizee.values]
        if lab == 'match':
            lab = 'local event'
        ax.scatter(tt.mean_el.values, tt.len_thread.values, label=lab, s=size, c=colors[ind_lab])
        ax.scatter(np.NaN, np.NaN, color='none', label=' mm')
        ax.scatter(np.NaN, np.NaN, color='none', label=' mm')

    ax.legend(bbox_to_anchor=(0,1.02,1,0.2), loc='lower left', ncol=4, fontsize=FONT_SIZE-2)
    ax.set_xlabel('Mean number of impacted cells over time', fontsize=FONT_SIZE)
    ax.set_ylabel('Event duration ()', fontsize=FONT_SIZE)
    ax.loglog()
    for axis in [ax.xaxis, ax.yaxis]:
        formatter = ScalarFormatter()
        formatter.set_scientific(False)
        axis.set_major_formatter(formatter)
    ax.xaxis.set_tick_params(labelsize=10)
    ax.yaxis.set_tick_params(labelsize=10)
    fig.tight_layout()
    
    fig.savefig('events_pos_anomalies.png', dpi=400)
    
    fig, ax = plt.subplots(figsize=[6.7, 4.4], dpi=400)
    for ind_lab, lab in enumerate(['unknown', 'bank holiday', 'outage']):
        tt = st_clusters_neg[st_clusters_neg.label==lab]
        sizes = tt.groupby(['mean_el', 'len_thread']).count()['sign'].reset_index()
        tt['sizee'] = tt.apply(lambda x: sizes[(sizes.mean_el==x['mean_el']) & (sizes.len_thread==x['len_thread'])].sign.values[0], axis=1)
        size = [15 + 100 * math.log(x) for x in tt.sizee.values]
        ax.scatter(tt.mean_el.values, tt.len_thread.values, label=lab, s=size, c=colors[ind_lab])
        ax.scatter(np.NaN, np.NaN, color='none', label=' mm')
        ax.scatter(np.NaN, np.NaN, color='none', label=' mm')

    ax.legend(bbox_to_anchor=(0,1.02,1,0.2), loc='lower left', ncol=3, fontsize=FONT_SIZE-2)
    ax.set_xlabel('Mean number of impacted cells over time', fontsize=FONT_SIZE)
    ax.set_ylabel('Event duration', fontsize=FONT_SIZE)
    ax.loglog()
    for axis in [ax.xaxis, ax.yaxis]:
        formatter = ScalarFormatter()
        formatter.set_scientific(False)
        axis.set_major_formatter(formatter)
    ax.xaxis.set_tick_params(labelsize=10)
    ax.yaxis.set_tick_params(labelsize=10)
    ax.set_xlim([0.9, 12])
    ax.set_ylim([0.6, 200])
    fig.tight_layout()
    
    fig.savefig('events_neg_anomalies.png', dpi=400)

def get_rarity_coefficient(mapping, st_clusters_pos, st_clusters_neg):
    APPS = [mapping.loc[mapping.id_port == port]['Application'].values[0] for port in ports_selected]

    colors = dict()
    list_category = ['Web', 'Download', 'CloudStorage', 'Others', 'Streaming', 'Chat', 'Mail']
    for color, cat in zip(plt.rcParams['axes.prop_cycle'].by_key()['color'], list_category):
        colors[cat] = color
    labels = list(colors.keys())
    handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]

    st_clusters_pos = st_clusters_pos.sort_values('mean_el', ascending=False)
    st_clusters_neg = st_clusters_neg.sort_values('mean_el', ascending=False)
    encod_st_clusters_pos = pd.DataFrame(index=st_clusters_pos.index.tolist(), columns=APPS)
    encod_st_clusters_pos['label'] = st_clusters_pos['label']
    encod_st_clusters_neg = pd.DataFrame(index=st_clusters_neg.index.tolist(), columns=APPS)
    encod_st_clusters_neg['label'] = st_clusters_neg['label']

    for index, row in encod_st_clusters_pos.iterrows():
        applis = st_clusters_pos.loc[index]['list_applis_all']
        for app in APPS:
            if app in applis:
                encod_st_clusters_pos.at[index, app] = 1      

    ll = st_clusters_pos.list_applis_all.values
    temp = pd.DataFrame({'app': [item for sublist in ll for item in sublist], 'val': [0]*len([item for sublist in ll for item in sublist])})
    temp = temp.groupby('app').count().reset_index()
    total = temp.sort_values(by='val', ascending=False)

    for ind_lab, lab in enumerate(['match', 'national event', 'service update']):
        df_label = pd.DataFrame(0, index=APPS, columns={'occurrence', 'prop'})
        for index, row in st_clusters_pos[st_clusters_pos.label == lab].iterrows():
            temp = pd.DataFrame({'app': row['list_applis_all'], 'occurrence': [0]*len(row['list_applis_all'])})
            temp = temp.groupby('app', as_index=False).count()
            temp['total'] = temp['app'].apply(lambda x: total[total.app==x].val.values[0])
            temp['prop'] = temp['occurrence'] / temp['total']
            temp = temp.set_index('app')
            df_label = df_label.add(temp, fill_value=0)
            
        df_label['prop'] = df_label['prop'].apply(lambda x: x/len(st_clusters_pos[st_clusters_pos.label == lab].values))
        df_label = df_label.sort_values('prop', ascending=False).dropna()
        if ind_lab != 0:
            fig, ax = plt.subplots(figsize=[6.7, 2.75], dpi=400)
        else:
            fig, ax = plt.subplots(figsize=[6.7, 3.2], dpi=400)
        ll = df_label.index.tolist()
        if 'Apple notification and push' in ll:
            ll[ll.index('Apple notification and push')] = 'Apple push'
        if 'Google+ content sharing + Drive' in ll:
            ll[ll.index('Google+ content sharing + Drive')] = 'Google+CDN'
        if 'Instagram Videos MP4' in ll:
            ll[ll.index('Instagram Videos MP4')] = 'Instagram Videos'
        if 'Google Play Store' in ll:
            ll[ll.index('Google Play Store')] = 'Play Store'
        if 'Facebook Streaming' in ll:
            ll[ll.index('Facebook Streaming')] = 'Facebook Stream'
        if 'HTTP Mail Microsoft' in ll:
            ll[ll.index('HTTP Mail Microsoft')] = 'Microsoft Mail'
        ax.bar(x=ll,
               height=df_label.prop.tolist(), color=[colors[mapping[mapping.Application==app].appli_desc.values[0]] for app in df_support[lab].sort_values(ascending=False).index])
        plt.xticks(rotation=90, ha='right')
        plt.text(-4, -0.004, r'$\rho_e$', fontsize=10)
        ax.set_ylim([0, 0.02])
        ax.yaxis.set_tick_params(labelsize=8)
        if ind_lab == 0:
            ax.legend(handles, labels, bbox_to_anchor=(0,1.02,1,0.2), loc='lower left', ncol=4, fontsize=10)
        fig.tight_layout()
        fig.savefig(lab + '_mean_rarity.png')

    for index, row in encod_st_clusters_neg.iterrows():
        applis = st_clusters_neg.loc[index]['list_applis_all']
        for app in APPS:
            if app in applis:
                encod_st_clusters_neg.at[index, app] = 1

    ll = st_clusters_neg.list_applis_all.values
    temp = pd.DataFrame({'app': [item for sublist in ll for item in sublist], 'val': [0]*len([item for sublist in ll for item in sublist])})
    temp = temp.groupby('app').count().reset_index()
    total = temp.sort_values(by='val', ascending=False)

    for ind_lab, lab in enumerate(['outage', 'bank holiday']):
        df_label = pd.DataFrame(0, index=APPS, columns={'occurrence', 'prop'})
        for index, row in st_clusters_neg[st_clusters_neg.label == lab].iterrows():
            temp = pd.DataFrame({'app': row['list_applis_all'], 'occurrence': [0]*len(row['list_applis_all'])})
            temp = temp.groupby('app', as_index=False).count()
            temp['total'] = temp['app'].apply(lambda x: total[total.app==x].val.values[0])
            temp['prop'] = temp['occurrence'] / temp['total']
            temp = temp.set_index('app')
            df_label = df_label.add(temp, fill_value=0)
            
        df_label['prop'] = df_label['prop'].apply(lambda x: x/len(st_clusters_neg[st_clusters_neg.label == lab].values))
        df_label = df_label.sort_values('prop', ascending=False).dropna()
        ll = df_label.index.tolist()
        if 'Apple notification and push' in ll:
            ll[ll.index('Apple notification and push')] = 'Apple push'
        if 'Google+ content sharing + Drive' in ll:
            ll[ll.index('Google+ content sharing + Drive')] = 'Google+CDN'
        if 'Instagram Videos MP4' in ll:
            ll[ll.index('Instagram Videos MP4')] = 'Instagram Videos'
        if 'Google Play Store' in ll:
            ll[ll.index('Google Play Store')] = 'Play Store'
        if 'Facebook Streaming' in ll:
            ll[ll.index('Facebook Streaming')] = 'Facebook Stream'
        if 'HTTP Mail Microsoft' in ll:
            ll[ll.index('HTTP Mail Microsoft')] = 'Microsoft Mail'
        fig, ax = plt.subplots(figsize=[6.7, 3.2], dpi=400)
        ax.bar(x=ll,
               height=df_label.prop.tolist(), color=[colors[mapping[mapping.Application==app].appli_desc.values[0]] for app in df_support[lab].sort_values(ascending=False).index])
        plt.xticks(rotation=90, ha='right')
        plt.text(-3.7, -0.0014, r'$\rho_e$', fontsize=10)
        ax.set_ylim([0, 0.006])
        ax.yaxis.set_tick_params(labelsize=8)
        ax.legend(handles, labels, bbox_to_anchor=(0,1.02,1,0.2), loc='lower left', ncol=4, fontsize=10)
        fig.tight_layout()
        
        fig.savefig(lab + '_mean_rarity.png')

def get_jaccard_similarity(df_anomaly, spatial_clusters_pos, spatial_clusters_neg):
    top_events = st_clusters_pos.sort_values('len_thread', ascending=False).iloc[:5]

    ll = st_clusters_pos.list_applis_all.values
    temp = pd.DataFrame({'app': [item for sublist in ll for item in sublist], 'val': [0]*len([item for sublist in ll for item in sublist])})
    temp = temp.groupby('app').count().reset_index()
    total = temp.sort_values(by='val', ascending=False)

    for ind, top_event in top_events.iterrows():
        sp_clusters_event = spatial_clusters_pos[spatial_clusters_pos.index.isin(top_event.indices)]
        sp_clusters_event = sp_clusters_event.sort_values('time_slot')
        sp_clusters_event['len'] = sp_clusters_event['list_cells'].apply(lambda x: len(x))
        
        zones = set()
        for index, row in sp_clusters_event.iterrows():
            for zone in row['list_cells']:
                zones.add(zone)
        instants = [str(x) for x in sp_clusters_event.time_slot.values]
        mat_jacc_distance_zones = {}
        mat_eucl_distance_zones = {}
        if len(zones) > 1:
            for zone in zones:
                for ind_inst, instant in enumerate(instants):
                    count_port = df_anomaly[(df_anomaly.place == zone) & (df_anomaly.time_slot == instant)].groupby(['port_app']).count()['attribute']
                    count_port = count_port.reset_index()
                    count_port['index'] = count_port['port_app'].apply(lambda port: mapping.loc[mapping.id_port == port]['Application'].values[0])
                    count_port = count_port[['index', 'attribute']]
                    if ind_inst == 0:
                        new_df = count_port
                    else:
                        new_df = pd.merge(new_df, count_port, on='index', how='outer')
                    new_df = new_df.rename(columns={'attribute': str(instant)})

                len_mat = len(new_df.columns[1:])
                new_df = new_df.rename(columns={'index': 'app'})
                mat_jacc_distance_zones[zone] = np.zeros((len_mat, len_mat))
                mat_eucl_distance_zones[zone] = np.zeros((len_mat, len_mat))
                for ind1, col1 in enumerate(instants):
                    for ind2, col2 in enumerate(instants):
                        mat_jacc_distance_zones[zone][ind1][ind2] = jaccard_similarity(new_df.dropna(subset=[col1]).app.unique(), new_df.dropna(subset=[col2]).app.unique())

            # for k, v in mat_jacc_distance_zones.ix in new_df.columns[1:]], y=[str(x) for x in new_df.columns[1:]], zmin=0, zmax=1, width=700, height=700)

highlights = {'common': [Label('2019-04-22 00:00:00', '2019-04-23 00:00:00', 'bank holiday', 'lundi de Pâques'),
              Label('2019-05-01 00:00:00', '2019-05-02 00:00:00',  'bank holiday', 'fête du travail'),
              Label('2019-05-08 00:00:00', '2019-05-09 00:00:00', 'bank holiday', 'Armistice 45'),
              Label('2019-05-30 00:00:00', '2019-05-31 00:00:00', 'bank holiday', 'Ascension'),
              Label('2019-04-15 19:00:00', '2019-04-16 00:00:00', 'national event', 'incendie Notre-Dame'),
              Label('2019-04-15 12:00:00', '2019-04-15 12:30:00', 'outage', 'Panne Orange 15 avril'),
              Label('2019-04-09 16:00:00', '2019-04-09 18:00:00', 'service update', 'Panne Orange 15 avril'),
              Label('2019-04-24 15:30:00', '2019-04-24 18:00:00', 'service update', 'Panne Orange 15 avril')],
             'stade_france': [Label('2019-03-25 17:00:00', '2019-03-26 02:00:00', 'match', 'France/Islande'),
              Label('2019-04-27 17:00:00', '2019-04-28 02:00:00', 'match', 'finale coupe de France'),
              Label('2019-05-12 17:00:00', '2019-05-13 02:00:00', 'match', 'Metallica'),
             Label('2019-05-18 17:00:00', '2019-05-19 02:00:00', 'match', 'Stars80')]}

MAIN_PATH = '/Users/agatheblaise/orange_datasets/'
DATES = ['2019-03-16 11:00:00', '2019-06-06']
PERIOD = pd.date_range(*DATES, freq='30min')
LIST_ATTS = ['s_nPacketUp', 's_nPacketDn', 'distinct_users']
FREQ = int(24*60/30*7)
LAG = 48*7
THRESHOLD = 3.5
INFLUENCE = 0

twoG_antennas = pd.read_csv('/Users/agatheblaise/orange_datasets/antennas/NORIA_CELLS_2G.csv', header=0, sep=';')
threeG_antennas = pd.read_csv('/Users/agatheblaise/orange_datasets/antennas/NORIA_CELLS_3G.csv', header=0, sep=';')
fourG_antennas = pd.read_csv('/Users/agatheblaise/orange_datasets/antennas/NORIA_CELLS_4G.csv', header=0, sep=',')

df_SDF = pd.read_csv('Saint-denis_data.csv', names=['PortApp', 'LocInfo', 'COORD_X', 'COORD_Y', 'NOM_SITE',
                                                  'TimeSlot', 's_nPacketUp', 's_nPacketDn', 's_Duration', 'distinct_users',
                                                 'nb_flows'], sep=';')

df_SDF = df_SDF.groupby(['NOM_SITE', 'COORD_X', 'COORD_Y', 'PortApp', 'TimeSlot'])['s_nPacketUp', 's_nPacketDn', 'distinct_users'].sum()
df_SDF = df_SDF.reset_index()
df_SDF = df_SDF[~df_SDF.NOM_SITE.isin(['CREVECOEUR', 'SURDENS_A86_PLEYEL', 'SAINT_GRATIEN_LES_JOLIVATS'])]
df_SDF = df_SDF[df_SDF.distinct_users > 5]
df_SDF['traffic'] = df_SDF['s_nPacketUp'] + df_SDF['s_nPacketDn']
gp = df_SDF.groupby(['PortApp'])['traffic'].sum()
gp = gp.sort_values(ascending=False)
gp = gp.reset_index()

ports_selected = gp.iloc[:40].PortApp.tolist()
df_SDF = df_SDF[df_SDF.PortApp.isin(ports_selected)]

mapping = pd.read_csv(MAIN_PATH + 'PortApp.csv', header=0)

df_anomaly = {}
df_anomaly = pd.read_csv('df_anomaly_wo_0.csv')
df_anomaly = df_anomaly[df_anomaly.port_app.isin(ports_selected)]
df_anomaly = df_anomaly[~df_anomaly.place.isin(['CREVECOEUR', 'SURDENS_A86_PLEYEL', 'SAINT_GRATIEN_LES_JOLIVATS'])] 
df_anomaly = df_anomaly[~df_anomaly.place.str.contains('TRIB')]
df_anomaly = df_anomaly[~df_anomaly.place.str.contains('ANX')]
df_anomaly = df_anomaly[df_anomaly.len_ts > 30]

# anomalies = {}
# anomalies['res'] = []
# for port_app in df_SDF.PortApp.unique():
#     df_SDF_1 = df_SDF[df_SDF.PortApp == port_app]
#     for site in df_SDF_1.NOM_SITE.unique():
#         df_SDF_2 = df_SDF_1[df_SDF_1.NOM_SITE == site]
#         COORD_X, COORD_Y = df_SDF_2.iloc[0].COORD_X, df_SDF_2.iloc[0].COORD_Y
#         df_SDF_2 = df_SDF_2.set_index('TimeSlot')
#         df_SDF_2.index = pd.DatetimeIndex(df_SDF_2.index)
#         df_SDF_2 = df_SDF_2.reindex(PERIOD, fill_value=0)
#         df_SDF_2 = df_SDF_2.loc[DATES[0]:DATES[1]]
#         df_SDF_2 = df_SDF_2.rename_axis('TimeSlot').reset_index()
#         df_SDF_2 = df_SDF_2.set_index('TimeSlot')
        
#         for ind_att, att in enumerate(LIST_ATTS):
#             result = thresholding_algo(df_SDF_2[att], lag=LAG, threshold=THRESHOLD, influence=INFLUENCE)
#             for loc_item, item in enumerate(result['signals']):
#                 if abs(item) > 0:
#                     anomalies['res'].append(Anomaly(df_SDF_2.iloc[int(loc_item)].name, port_app, att, 'residual', site, COORD_X, COORD_Y, item, result['len_ts'][loc_item]))

spatial_clusters_pos, spatial_clusters_neg, st_clusters_pos, st_clusters_neg = ({} for i in range(4))
dataframe = df_anomaly
points = {place: [get_X(place), get_Y(place)] for place in dataframe.place.unique()}
list_places = list(points.keys())
coord_points = list(points.values())
vor = Voronoi(coord_points)
regions, vertices = voronoi_finite_polygons_2d(vor)
neighbors = {}
for ind_1, region1 in enumerate(regions):
    neighbors[list_places[ind_1]] = set()
    for ind_2, region2 in enumerate(regions):
        if region1 != region2:
            common = [x for x in vertices[region1] if x in vertices[region2]]
            for com_el in  common:
                if vor.min_bound[0] < com_el[0] < vor.max_bound[0] and vor.min_bound[1] < com_el[1] < vor.max_bound[1]:
                    neighbors[list_places[ind_1]].add(list_places[ind_2])
                    break

in_group_T_C_iqr_pos, in_group_T_C_iqr_neg = get_groups_signed(dataframe, neighbors)
list_prop_clusters = {}
list_prop_clusters['pos'] = get_prop_clusters(neighbors, in_group_T_C_iqr_pos)
list_prop_clusters['neg'] = get_prop_clusters(neighbors, in_group_T_C_iqr_neg)

spatial_clusters_pos = pd.DataFrame([o.__dict__ for o in list_prop_clusters['pos']])
spatial_clusters_pos = spatial_clusters_pos.sort_values(by='time_slot')
dates = [spatial_clusters_pos['time_slot'].values[0], spatial_clusters_pos['time_slot'].values[-1]]
period = pd.date_range(*dates, freq='30min')
all_dates = pd.DataFrame(period, columns=['time_slot'])
all_dates['list_cells'] = [set()] * len(all_dates)
spatial_clusters_pos = pd.merge(spatial_clusters_pos, all_dates, how='outer', on='time_slot')
spatial_clusters_pos = spatial_clusters_pos.sort_values(by='time_slot')
spatial_clusters_pos['list_cells'] = spatial_clusters_pos.apply(lambda x: x.list_cells_x if str(x.list_cells_x) != 'nan' else {}, axis=1)

spatial_clusters_neg = pd.DataFrame([o.__dict__ for o in list_prop_clusters['neg']])
spatial_clusters_neg = spatial_clusters_neg.sort_values(by='time_slot')
dates = [spatial_clusters_neg['time_slot'].values[0], spatial_clusters_neg['time_slot'].values[-1]]
period = pd.date_range(*dates, freq='30min')
all_dates = pd.DataFrame(period, columns=['time_slot'])
all_dates['list_cells'] = [set()] * len(all_dates)
spatial_clusters_neg = pd.merge(spatial_clusters_neg, all_dates, how='outer', on='time_slot')
spatial_clusters_neg = spatial_clusters_neg.sort_values(by='time_slot')
spatial_clusters_neg['list_cells'] = spatial_clusters_neg.apply(lambda x: x.list_cells_x if str(x.list_cells_x) != 'nan' else {}, axis=1)

current_events_pos = []
for index, row in spatial_clusters_pos.iterrows():
    common_zone = ''
    for zone in row.list_cells:
        if not common_zone:
            for ind_ev, ev in enumerate(current_events_pos):
                if row.time_slot == ev.end_time + np.timedelta64(30, 'm'):
                    if zone in ev.list_zones[-1]:
                        common_zone = zone
                        current_events_pos[ind_ev].update_list_zones(index, row.list_cells, row.time_slot)
                        break
    if common_zone == '':
        current_events_pos.append(Event_signed(row.time_slot, index, row.list_cells, '+'))

current_events_neg = []
for index, row in spatial_clusters_neg.iterrows():
    common_zone = ''
    for zone in row.list_cells:
        if not common_zone:
            for ind_ev, ev in enumerate(current_events_neg):
                if row.time_slot == ev.end_time + np.timedelta64(30, 'm'):
                    if zone in ev.list_zones[-1]:
                        common_zone = zone
                        current_events_neg[ind_ev].update_list_zones(index, row.list_cells, row.time_slot)
                        break
    if common_zone == '':
        current_events_neg.append(Event_signed(row.time_slot, index, row.list_cells, '-'))

st_clusters_pos = pd.DataFrame([o.__dict__ for o in current_events_pos])
st_clusters_pos = st_clusters_pos.reset_index()
spatial_clusters_pos = spatial_clusters_pos[spatial_clusters_pos.list_cells != {}]

st_clusters_neg = pd.DataFrame([o.__dict__ for o in current_events_neg])
st_clusters_neg = st_clusters_neg.reset_index()
spatial_clusters_neg = spatial_clusters_neg[spatial_clusters_neg.list_cells != {}]

st_clusters_pos['len_thread'] = st_clusters_pos.list_zones.apply(lambda x: str(x).count('{'))
st_clusters_neg['len_thread'] = st_clusters_neg.list_zones.apply(lambda x: str(x).count('{'))
st_clusters_pos['mean_el'] = st_clusters_pos.list_zones.apply(lambda x: np.mean([len(y) for y in x]))
st_clusters_neg['mean_el'] = st_clusters_neg.list_zones.apply(lambda x: np.mean([len(y) for y in x]))
st_clusters_pos['n_ano'] = st_clusters_pos.list_zones.apply(lambda x: str(x).count(','))
st_clusters_neg['n_ano'] = st_clusters_neg.list_zones.apply(lambda x: str(x).count(','))

st_clusters_pos['label'] = st_clusters_pos.apply(label_event, axis=1)
st_clusters_neg['label'] = st_clusters_neg.apply(label_event, axis=1)

st_clusters_pos['start_time'] = pd.to_datetime(st_clusters_pos['start_time'])
st_clusters_neg['end_time'] = pd.to_datetime(st_clusters_neg['end_time'])
dataframe['time_slot'] = pd.to_datetime(dataframe['time_slot'])

st_clusters_pos['list_applis_all'] = st_clusters_pos.apply(get_list_applis_all, args=(dataframe,), axis=1)
st_clusters_neg['list_applis_all'] = st_clusters_neg.apply(get_list_applis_all, args=(dataframe,), axis=1)

# get_voronoi_diagram(df_SDF)
# example_decomp_ts(df_SDF)
plot_map_anomalies(df_anomaly, spatial_clusters_pos, spatial_clusters_neg)
spatiotemporal_events_visualization(st_clusters_pos, st_clusters_neg)
get_rarity_coefficient(df_anomaly, spatial_clusters_pos, spatial_clusters_neg)
get_jaccard_similarity(df_anomaly, spatial_clusters_pos, spatial_clusters_neg)
