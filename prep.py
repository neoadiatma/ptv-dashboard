# prep.py
import os, zipfile, requests, json
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, box
import topojson as tp

os.makedirs('data', exist_ok=True)

def download_file(url, dest):
    r = requests.get(url)
    with open(dest, 'wb') as f:
        f.write(r.content)

def clean_name(name):
    if pd.isna(name):
        return ''
    return name.strip().upper().replace(' RAILWAY STATION', '').replace(' STATION', '')

# --- Download annual station file ---
station_annual_url = "https://opendata.transport.vic.gov.au/dataset/annual-metropolitan-train-station-patronage-station-entries/resource/c9507eb5-aa48-4a43-aa09-c10a24d1f2fe/download/annual_metropolitan_train_station_patronage.csv"
print("Downloading annual station patronage ...")
download_file(station_annual_url, 'annual_station_raw.csv')

# Extract GTFS (assuming gtfs.zip exists in working directory)
if not os.path.exists('gtfs.zip'):
    raise FileNotFoundError("gtfs.zip not found. Please download it first.")
with zipfile.ZipFile('gtfs.zip') as z:
    z.extractall('gtfs')
nested_zip = 'gtfs/2/google_transit.zip'
if os.path.exists(nested_zip):
    with zipfile.ZipFile(nested_zip) as z:
        z.extractall('gtfs')

# Load GTFS tables
stops = pd.read_csv('gtfs/stops.txt', dtype=str)
shapes = pd.read_csv('gtfs/shapes.txt', dtype=str)
trips = pd.read_csv('gtfs/trips.txt', dtype=str)
routes = pd.read_csv('gtfs/routes.txt', dtype=str)
stop_times = pd.read_csv('gtfs/stop_times.txt', dtype=str, low_memory=False)

# Convert numeric columns
for col in ['shape_pt_lat', 'shape_pt_lon', 'shape_pt_sequence']:
    shapes[col] = pd.to_numeric(shapes[col])
routes['route_type'] = pd.to_numeric(routes['route_type'])
stops['location_type'] = pd.to_numeric(stops['location_type'], errors='coerce').fillna(0)

# Train stops (parent stations)
train_stops = stops[stops['location_type'] == 1].copy()

# --- Train line geometries ---
trips_routes = trips.merge(routes[['route_id', 'route_short_name', 'route_long_name', 'route_type']], on='route_id')
train_routes = routes[routes['route_type'].isin([0,1,2])]
train_trips = trips_routes[trips_routes['route_id'].isin(train_routes['route_id'])]
shapes_sorted = shapes.sort_values(['shape_id', 'shape_pt_sequence'])
shape_geoms = shapes_sorted.groupby('shape_id')[['shape_pt_lon', 'shape_pt_lat']].apply(
    lambda g: LineString(g.values)
).reset_index(name='geometry')
shape_geoms = gpd.GeoDataFrame(shape_geoms, geometry='geometry', crs='EPSG:4326')
trip_shapes = train_trips[['route_short_name', 'shape_id']].drop_duplicates()
line_geoms = shape_geoms.merge(trip_shapes, on='shape_id')
line_geoms['geometry'] = line_geoms['geometry'].simplify(0.001)
line_geoms = line_geoms.dissolve(by='route_short_name').reset_index()
line_geoms[['route_short_name', 'geometry']].to_file('data/train_lines.json', driver='GeoJSON')

# --- Stations GeoDataFrame ---
stations_gdf = gpd.GeoDataFrame(
    train_stops,
    geometry=gpd.points_from_xy(pd.to_numeric(train_stops.stop_lon), pd.to_numeric(train_stops.stop_lat)),
    crs='EPSG:4326'
)
stations_gdf = stations_gdf[['stop_id', 'stop_name', 'geometry']]
stations_gdf['clean_name'] = stations_gdf['stop_name'].apply(clean_name)

# --- Annual station patronage (long format) ---
annual_df = pd.read_csv('annual_station_raw.csv')
annual_df['Fin_year'] = annual_df['Fin_year'].astype(str).str.strip()
latest_year = annual_df['Fin_year'].max()
print(f"Using most recent financial year: {latest_year}")
annual_latest = annual_df[annual_df['Fin_year'] == latest_year].copy()
annual_latest['clean_name'] = annual_latest['Stop_name'].apply(clean_name)
annual_latest['entries'] = pd.to_numeric(annual_latest['Pax_annual'], errors='coerce').fillna(0)

stations_merged = stations_gdf.merge(annual_latest[['clean_name', 'entries']], on='clean_name', how='left')
stations_merged['entries'] = stations_merged['entries'].fillna(0)

# Dominant line per station
# Create a mapping from each stop_id (including child platforms) to its parent station
stops_all = pd.read_csv('gtfs/stops.txt', dtype=str)
# If a stop has a parent_station, use it; otherwise the stop itself is its own parent
stops_all['parent_id'] = stops_all['parent_station'].fillna(stops_all['stop_id'])
child_to_parent = stops_all.set_index('stop_id')['parent_id'].to_dict()

# Map stop_times.stop_id -> parent station ID
stop_times['parent_stop_id'] = stop_times['stop_id'].map(child_to_parent)

stop_times['trip_id'] = stop_times['trip_id'].astype(str)
trips_routes['trip_id'] = trips_routes['trip_id'].astype(str)

# Count trips per parent station per route
stop_to_line = stop_times.merge(
    trips_routes[['trip_id', 'route_short_name']], on='trip_id'
)
stop_line_counts = stop_to_line.groupby(
    ['parent_stop_id', 'route_short_name']
).size().reset_index(name='count')

# Get the dominant line for each parent station
dominant_line = stop_line_counts.loc[
    stop_line_counts.groupby('parent_stop_id')['count'].idxmax()
][['parent_stop_id', 'route_short_name']]
dominant_line.rename(columns={'parent_stop_id': 'stop_id'}, inplace=True)

stations_merged = stations_merged.merge(dominant_line, on='stop_id', how='left')
stations_merged['route_short_name'] = stations_merged['route_short_name'].fillna('Other')

# Save stations for Vega-Lite
stations_merged['lon'] = stations_merged.geometry.x
stations_merged['lat'] = stations_merged.geometry.y
stations_out = stations_merged[['stop_id', 'stop_name', 'lon', 'lat', 'entries', 'route_short_name']]
stations_out.to_json('data/stations.json', orient='records')

# Top 10 stations
top10 = stations_out.nlargest(10, 'entries')[['stop_name', 'entries']]
top10.to_csv('data/top10_stations.csv', index=False)

# Line total entries
line_entries = stations_merged.groupby('route_short_name')['entries'].sum().reset_index()
line_entries.columns = ['Line', 'TotalEntries']
line_entries.to_csv('data/line_entries.csv', index=False)

# AM vs PM peak scatter
annual_latest['AM_peak'] = pd.to_numeric(annual_latest['Pax_AM_peak'], errors='coerce').fillna(0)
annual_latest['PM_peak'] = pd.to_numeric(annual_latest['Pax_PM_peak'], errors='coerce').fillna(0)
annual_latest['Stop_name_clean'] = annual_latest['Stop_name'].apply(clean_name)
am_pm = annual_latest[['Stop_name_clean', 'AM_peak', 'PM_peak']]
am_pm.to_csv('data/am_pm_scatter.csv', index=False)

# Top 10 peak breakdown
top10_ids = stations_out.nlargest(10, 'entries')['stop_id']
top10_detail = stations_merged[stations_merged['stop_id'].isin(top10_ids)].copy()
peak_cols = ['Pax_AM_peak', 'Pax_interpeak', 'Pax_PM_peak', 'Pax_PM_late']
top10_detail = top10_detail.merge(
    annual_latest[['clean_name'] + peak_cols], on='clean_name', how='left'
)
for col in peak_cols:
    top10_detail[col] = pd.to_numeric(top10_detail[col], errors='coerce').fillna(0)
top10_peak = top10_detail.melt(
    id_vars=['stop_name'], value_vars=peak_cols,
    var_name='Peak', value_name='Passengers'
)
top10_peak['Peak'] = top10_peak['Peak'].str.replace('Pax_', '')
top10_peak.to_csv('data/top10_peak.csv', index=False)

# Hourly frequency - use known train line names instead of route_type filtering
def hour_from_gtfs(time_str):
    try:
        parts = time_str.split(':')
        return int(parts[0]) % 24
    except:
        return None

stop_times['arrival_hour'] = stop_times['arrival_time'].apply(hour_from_gtfs)
stop_times = stop_times.dropna(subset=['arrival_hour'])

# Known metropolitan train lines (same as the map dropdown)
train_line_names = [
    "Alamein", "Belgrave", "Craigieburn", "Cranbourne",
    "Frankston", "Glen Waverley", "Hurstbridge", "Lilydale",
    "Mernda", "Pakenham", "Sandringham", "Sunbury",
    "Upfield", "Werribee", "Williamstown"
]

# Filter trips_routes to only those lines
trips_routes_train = trips_routes[trips_routes['route_short_name'].isin(train_line_names)].copy()

# Merge stop_times with train trips (including parent_stop_id for later use)
weekday_trips = trips_routes_train.merge(
    stop_times[['trip_id', 'stop_id', 'arrival_hour', 'parent_stop_id']],
    on='trip_id'
)

# Aggregate hourly frequency
hourly_freq = weekday_trips.groupby(['route_short_name', 'arrival_hour']).size().reset_index(name='trips')
hourly_freq.rename(columns={'route_short_name': 'Line', 'arrival_hour': 'Hour'}, inplace=True)
hourly_freq.to_csv('data/hourly_frequency.csv', index=False)

print(f"Hourly frequency rows: {len(hourly_freq)}")

# Regions
cbd_names = ['FLINDERS STREET', 'SOUTHERN CROSS', 'MELBOURNE CENTRAL', 'PARLIAMENT', 'FLAGSTAFF']
stations_merged['Region'] = 'Outer'
stations_merged.loc[stations_merged['clean_name'].isin(cbd_names), 'Region'] = 'CBD'
inner_mask = (
    stations_merged['lon'].between(144.9, 145.1) &
    stations_merged['lat'].between(-37.9, -37.75) &
    (stations_merged['Region'] != 'CBD')
)
stations_merged.loc[inner_mask, 'Region'] = 'Inner'
region_patronage = stations_merged.groupby('Region')['entries'].sum().reset_index()
region_patronage.to_csv('data/region_patronage.csv', index=False)

# Annual change (long format)
annual_sorted = annual_df.sort_values(['Stop_name', 'Fin_year'])
def compute_change(group):
    if len(group) >= 2:
        group = group.sort_values('Fin_year')
        last = pd.to_numeric(group.iloc[-1]['Pax_annual'], errors='coerce')
        prev = pd.to_numeric(group.iloc[-2]['Pax_annual'], errors='coerce')
        if pd.notna(last) and pd.notna(prev):
            return last - prev
    return 0

change_series = annual_sorted.groupby('Stop_name', group_keys=False).apply(compute_change).reset_index(name='change')
change_df = change_series[change_series['change'] != 0]
change_df = change_df.nlargest(20, 'change')
change_df.to_csv('data/station_change.csv', index=False)

# Services scatter
stop_services = weekday_trips.groupby('parent_stop_id').size().reset_index(name='services_per_day')
stop_services.rename(columns={'parent_stop_id': 'stop_id'}, inplace=True)
stations_merged = stations_merged.merge(stop_services, on='stop_id', how='left')
stations_merged['services_per_day'] = stations_merged['services_per_day'].fillna(0)
scatter_data = stations_merged[['stop_name', 'entries', 'services_per_day']]
scatter_data.to_csv('data/services_scatter.csv', index=False)

# --- Weekday vs weekend patronage for top 20 stations ---
top20_ids = stations_out.nlargest(20, 'entries')['stop_id']
weekend_detail = stations_merged[stations_merged['stop_id'].isin(top20_ids)].copy()
# Get weekday/Sat/Sun from annual_latest (already has clean_name)
weekend_detail = weekend_detail.merge(
    annual_latest[['clean_name', 'Pax_weekday', 'Pax_Saturday', 'Pax_Sunday']],
    on='clean_name', how='left'
)
for col in ['Pax_weekday', 'Pax_Saturday', 'Pax_Sunday']:
    weekend_detail[col] = pd.to_numeric(weekend_detail[col], errors='coerce').fillna(0)

# Melt to long form for grouped bar
weekend_melt = weekend_detail.melt(
    id_vars=['stop_name'], value_vars=['Pax_weekday', 'Pax_Saturday', 'Pax_Sunday'],
    var_name='DayType', value_name='Passengers'
)
weekend_melt['DayType'] = weekend_melt['DayType'].str.replace('Pax_', '')
weekend_melt.to_csv('data/weekday_weekend.csv', index=False)

# # LGA background (bounding box)
# print("Creating LGA background...")
# melb_bbox = box(144.5, -38.5, 145.5, -37.5)
# melb_poly = gpd.GeoDataFrame({'name': ['Greater Melbourne']}, geometry=[melb_bbox], crs='EPSG:4326')
# topo = tp.Topology(melb_poly, prequantize=False)
# topo.to_json('data/lga_melbourne.json')
print("Done.")