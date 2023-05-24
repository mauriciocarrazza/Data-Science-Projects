import pandas as pd
import numpy as np
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df = pd.read_csv('I:\My Drive\Data Projects\Customer Segmentation\data_boa.csv', encoding='latin-1')

#### 1. DATA PRE-PROCESSING ####
# Lowercase all names to avoid getting 2 unique same customers
# DESCRIPTION Column: Contains either human or corporation names
df['DESCRIPTION'] = df['DESCRIPTION'].str.lower()

# Remove null names and zip codes to only have data with identifiable customers
# ZIP CODE Column: contains the zip code in numeric format.
df = df.dropna(subset=['DESCRIPTION', 'ZIP CODE'])

# Obtain a unique customer column by concatenating DESCRIPTION and ZIP CODE columns
df['CUSTOMER'] = df['DESCRIPTION'] + " " + df['ZIP CODE']


#### 2. CALCULATE RFM METRICS ####
# (For this dataset, we dont have Monetary Value)
# Create a hipothetical snapshot_date as we we're doing the analysis one day after the last purchase
df['ORDER DATE'] = pd.to_datetime(df['ORDER DATE'])
snapshot_date = max(df['ORDER DATE']) + dt.timedelta(days=1)

# Calculate Recency and Frequency
# Recency: Days since the last purchase
# Frequency: # Of purchases 
df_rf = df.groupby(['CUSTOMER']).agg({
    'ORDER DATE': lambda x: (snapshot_date - x.max()).days,
    'SKU': 'count'
})
# Rename columns for easier interpretation
df_rf.rename(columns = {'ORDER DATE':'Recency', 'SKU':'Frequency'}, inplace=True)
#print(df_rf.head())
#print(df_rf.describe())

# View data to see if we can remove outliers
#plt.title('Customers'); plt.xlabel('Recency [days]'); plt.ylabel('Frequency [# of purchases]')
#sns.scatterplot(data = df_rf, x = 'Recency', y = 'Frequency')
#plt.show()
# -> 2 Clear Outliers by Frequency

# (Extra) Lets store the name of customers with >= 11 purchases:
frequent_buyers = df_rf[df_rf['Frequency'] >= 11].sort_values('Frequency', ascending=False)
#print(frequent_buyers)

# Remove customers with > 11 purchases
#print('Number of customers before outliers remove: ', df_rf.shape[0])
df_rf = df_rf[df_rf['Frequency'] < 11]
#print('Number of customers after outliers remove: ', df_rf.shape[0])
#-> Only 11 Removed (from 72k)

# View data with removed outliers
#plt.title('Customers'); plt.xlabel('Recency [days]'); plt.ylabel('Frequency [# of purchases]')
#sns.scatterplot(data = df_rf, x = 'Recency', y = 'Frequency')
#plt.show()

### 3. PRE-PROCESSING THE DATA FOR K-MEANS CLUSTERING ###
# We will use K-Means Clustering instead of pre-defined segments for customer segmentation.
# What is difference between segmentation and clustering?
# Both complement each other:
# - Segmentation involves human-defined groupings 
# - Clustering involves ML-powered groupings

# Key K-Means Assumptions: 
# - Symmetric distributions of variables
# - Variables with same mean
# - Variables with same variance

# Check skewness
#sns.displot(df_rf['Recency']); plt.show()
# -> Recency: Right skewed
#sns.displot(df_rf['Frequency']); plt.show()
# -> Frequency: Drastically Right skewed

# Check mean and variance
#print('Main statistics of our data:\n ', df_rf.describe())
# -> Clear differences between mean and variance

# Unskew the data
df_rf_log = np.log(df_rf)
#sns.displot(df_rf_log['Recency']); plt.show()
#sns.displot(df_rf_log['Frequency']); plt.show()

# Normalize the variables
scaler = StandardScaler()
scaler.fit(df_rf_log)
df_rf_normalized = scaler.transform(df_rf_log)
#print(df_rf_normalized)
#print('mean: ', np.mean(df_rf_normalized))
#print('std: ', np.std(df_rf_normalized))

### 4. RUNNING K-MEANS ###
## Choosing Number of Clusters: Elbow criterion
# Fit KMeans and calculate SSE for each *k*
sse = {}
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=1)
    kmeans.fit(df_rf_normalized)
    sse[k] = kmeans.inertia_ # sum of squared distances to closest cluster center

# Plot SSE for each *k*
#plt.title('The Elbow Method')
#plt.xlabel('k'); plt.ylabel('SSE')
#sns.pointplot(x=list(sse.keys()), y=list(sse.values()))
#plt.show()
# -> Optimal number: 3 clusters

## Compute k-means using 3 clusters
kmeans = KMeans(n_clusters=3, random_state=1)
# Compute k-means clustering on pre-processed data
kmeans.fit(df_rf_normalized)
# Extract cluster labels from labels_ attribute
cluster_labels = kmeans.labels_
# Create a cluster label column in the original DataFrame
df_rf_k3 = df_rf.assign(Cluster = cluster_labels)
# Calculate average RF values and size for each cluster
df_rf_k3_stats =  df_rf_k3.groupby(['Cluster']).agg({
'Recency': 'mean',
'Frequency': ['mean', 'count']
}).round(0)
#print(df_rf_k3_stats)
# -> Group 0 (51k: 71%): Only 1 purchase and longest time without buying
# -> Group 1 (2k: 3%): More than 1 purchase
# -> Group 2 (19k: 26%): Only 1 purchase and shortest time without buying

## Plot customer groups (viable cuz only 2 variables -> 2D scatterplot)
#plt.title('Three Customer Groups'); plt.xlabel(' Recency [days]'); plt.ylabel(' Frequency [# of purchases]')
#sns.scatterplot(data = df_rf_k3, x = 'Recency', y = 'Frequency', hue = 'Cluster')
#plt.show()

## Understand and compare segments (Snake plots)
# Transform df_rf_normalized as DataFrame and add a Cluster column
df_rf_normalized = pd.DataFrame(df_rf_normalized,
    index=df_rf.index,
    columns=df_rf.columns)
df_rf_normalized['Cluster'] = df_rf_k3['Cluster']
#print(df_rf_normalized)
# Melt the data into a long format so RF values and metric names are stored in 1 column each
df_melt = pd.melt(df_rf_normalized.reset_index(),
    id_vars=['CUSTOMER', 'Cluster'],
    value_vars=['Recency', 'Frequency'],
    var_name='Attribute',
    value_name='Value')
#print(df_melt)
# Visualize in a snake plot
#plt.title('Snake plot of standardized variables')
#sns.lineplot(x="Attribute", y="Value", hue='Cluster', data=df_melt)
#plt.show()

## Identify relative importance of each segment's attribute (heatmap)
cluster_avg = df_rf_k3.groupby(['Cluster']).mean()
population_avg = df_rf_k3.mean()
relative_imp = cluster_avg / population_avg - 1
#print(relative_imp.round(2))
# As a ratio moves away from 0, attribute importance for a segment (relative to total pop.) increases
#-> Frequency is by far the most important attribute for the Segment #1
#-> And recency for the Segment #2 and #0.
# Plot a heatmap
#plt.figure(figsize=(4, 2))
#plt.title('Relative importance of attributes')
#sns.heatmap(data=relative_imp, annot=True, fmt='.2f', cmap='RdYlGn')
#plt.show()

### X. CONCLUSIONS AND RECOMENDATIONS ###
# - Get the price per purchase data so we can use Monetary Value and get more meaningful
# Customer segment description
# - Only 2k customers (from ~70k) bought more than 1 time.
# - Frequency shows an skewed distribution even after transformation with log, 
# which implies that another transformation method should be used