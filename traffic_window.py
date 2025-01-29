import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
# from interpret.glassbox import ExplainableBoostingRegressor
from sklearn.model_selection import train_test_split
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.tree import DecisionTreeClassifier, plot_tree



# Load the dataset
# data = pd.read_csv('data\Metro_Interstate_Traffic_Volume.csv')
data = pd.read_csv('data/Metro_Interstate_Traffic_Volume.csv')

# print( data.shape   )
# print( data.head() )

## columns and data types
#print( data.info() )

# ## statistical summary of the dataset
#print( data.describe() )

### convert date_time to datetime object
data['date_time'] = pd.to_datetime(data['date_time'])

data['hour'] = data['date_time'].dt.hour
data['day_of_week'] = data['date_time'].dt.dayofweek
data['month'] = data['date_time'].dt.month

################### TASK 4 ###################
# Check for holidays
# Categorize into Weekday, Weekend, and Holiday
"""
data['traffic_type'] = 'Weekday'
data.loc[data['day_of_week'] >= 5, 'traffic_type'] = 'Weekend'
data.loc[data['holiday'].notna(), 'traffic_type'] = 'Holiday'  # Handle NaN

# Group by hour and traffic type
hourly_traffic = data.groupby(['hour', 'traffic_type'])['traffic_volume'].agg(['mean', 'std']).reset_index()

# Ensure all hours are included for each traffic type
all_hours = pd.DataFrame({'hour': np.arange(24)})
hourly_traffic = hourly_traffic.pivot(index='hour', columns='traffic_type', values=['mean', 'std'])

# Reset index and rename columns properly
hourly_traffic = hourly_traffic.stack().reset_index()
hourly_traffic.columns = ['hour', 'traffic_type', 'mean', 'std']

# Calculate mean traffic volume for axhline
mean_values = hourly_traffic.groupby('traffic_type')['mean'].mean()

# Define colors and styles
colors = {'Holiday': 'blue', 'Weekday': 'red', 'Weekend': 'orange'}
line_styles = {'Holiday': 'dashed', 'Weekday': 'solid', 'Weekend': 'solid'}

# Plot setup
plt.figure(figsize=(12, 6))
sns.set_style("whitegrid")

# Plot each category with confidence interval
for category in ['Holiday', 'Weekday', 'Weekend']:
    subset = hourly_traffic[hourly_traffic['traffic_type'] == category]
    
    if not subset.empty:
        sns.lineplot(x='hour', y='mean', data=subset, label=category, color=colors[category], linestyle=line_styles[category])
        plt.fill_between(subset['hour'], subset['mean'] - subset['std'], subset['mean'] + subset['std'], 
                         alpha=0.2, color=colors[category])

# Add mean reference lines
for category in ['Holiday', 'Weekday', 'Weekend']:
    if category in mean_values:
        plt.axhline(y=mean_values[category], linestyle=line_styles[category], color=colors[category], linewidth=1.5, label=f"{category} Mean")

# Labels and legend
plt.title("Traffic Volume Trends by Hour", fontsize=16)
plt.xlabel("Hour of the Day", fontsize=12)
plt.ylabel("Traffic Volume", fontsize=12)
plt.legend()
plt.show()
"""



################### TASK 5 ###################

""""
# Plot traffic volume trends by hour
# Define seasons based on month
season_mapping = {
    12: 'Winter', 1: 'Winter', 2: 'Winter',
    3: 'Spring', 4: 'Spring', 5: 'Spring',
    6: 'Summer', 7: 'Summer', 8: 'Summer',
    9: 'Fall', 10: 'Fall', 11: 'Fall'
}
data['season'] = data['month'].map(season_mapping)

# Group by hour and season
seasonal_traffic = data.groupby(['hour', 'season'])['traffic_volume'].agg(['mean', 'std']).reset_index()

# Calculate mean traffic volume for each season (for axhline)
mean_values = seasonal_traffic.groupby('season')['mean'].mean()

# Define colors
colors = {'Winter': 'blue', 'Spring': 'green', 'Summer': 'red', 'Fall': 'orange'}

# Plot setup
plt.figure(figsize=(12, 6))
sns.set_style("whitegrid")

# Plot each season with confidence interval
for season in ['Winter', 'Spring', 'Summer', 'Fall']:
    subset = seasonal_traffic[seasonal_traffic['season'] == season]
    
    if not subset.empty:
        sns.lineplot(x='hour', y='mean', data=subset, label=season, color=colors[season])
        plt.fill_between(subset['hour'], subset['mean'] - subset['std'], subset['mean'] + subset['std'], 
                         alpha=0.2, color=colors[season])

# Add mean reference lines
for season in ['Winter', 'Spring', 'Summer', 'Fall']:
    plt.axhline(y=mean_values[season], linestyle='dashed', color=colors[season], linewidth=1.5
    #label=f"{season} Mean"
    )

# Labels and legend
plt.title("Traffic Volume Trends by Season", fontsize=16)
plt.xlabel("Hour of the Day", fontsize=12)
plt.ylabel("Traffic Volume", fontsize=12)
plt.legend()
plt.show()
"""



################### TASK 6 ###################
"""
num_cols = data.select_dtypes(include=['number']).columns

data_minmax = data.copy()
data_standard = data.copy()

# Min-Max Scaling -- scales between 0 and 1
scaler_minmax = MinMaxScaler()
data_minmax[num_cols] = scaler_minmax.fit_transform(data_minmax[num_cols])

# Standard scaler -- scales to mean 0 and variance 1
scaler_standard = StandardScaler()
data_standard[num_cols] = scaler_standard.fit_transform(data_standard[num_cols])

# Save the normalized datasets
data_minmax.to_csv('Metro_Interstate_Traffic_Volume_MinMax.csv', index=False)
data_standard.to_csv('Metro_Interstate_Traffic_Volume_Standardized.csv', index=False)


print("Min-Max Normalized Data:")
print(data_minmax[:3].head())

print("\nStandardized Data:")
print(data_standard[:3].head())
"""



#### Task 8 #####
# Convert categorical features using Label Encoding
"""
categorical_cols = ['holiday', 'weather_main', 'weather_description']

for col in categorical_cols:
    data[col] = LabelEncoder().fit_transform( data[col].astype( str ) )

# extract features
data.drop(columns=['date_time'], inplace=True)

# Select features and target
X = data.drop(columns=['traffic_volume'])
y = data['traffic_volume']

# Normalize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train EBM model
ebm = ExplainableBoostingRegressor()
ebm.fit(X_train, y_train)


# Extract and plot feature importance
# Extract feature names and importance scores
feature_names = list(X.columns)  # Use original column names
feature_importances = ebm.term_importances() # Correct way to get EBM feature importance

# Ensure feature names match the number of importance scores
if len(feature_importances) > len(feature_names):
    feature_names.extend([f"Interaction {i}" for i in range(len(feature_importances) - len(feature_names))])

# Plot feature importance
plt.figure(figsize=(10, 5))
plt.barh(feature_names, feature_importances, color='blue')
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance Using Explainable Boosting Machine (EBM)")
plt.show()
"""



#### Task 9 #####
# Set 'date_time' as the index
"""
data.set_index('date_time', inplace=True)

# Resample data to daily frequency (sum of traffic volume per day)
daily_data = data['traffic_volume'].resample('D').sum()

# Apply seasonal decomposition
decomposition = seasonal_decompose(daily_data, model='additive', period=7)  # Weekly seasonality

# Plot the decomposed time series
plt.figure(figsize=(12, 8))

plt.subplot(4, 1, 1)
plt.plot(daily_data, label='Original', color='black')
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(decomposition.trend, label='Trend', color='blue')
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(decomposition.seasonal, label='Seasonality', color='green')
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(decomposition.resid, label='Residuals', color='red')
plt.legend()

plt.tight_layout()
plt.show()
"""



#### Task 10 #####
# Decision Tree Classifier
data.drop( columns=['date_time'], inplace=True )


# Encode categorical variables
categorical_cols = [ 'holiday', 'weather_main', 'weather_description' ]
for col in categorical_cols:
    data[col] = LabelEncoder().fit_transform( data[col].astype( str ) )


# Define binary classification target based on median traffic volume
threshold = data['traffic_volume'].median()
data['traffic_class'] = np.where( data['traffic_volume'] > threshold, 1, 0 )  # 1 = High Traffic, 0 = Low Traffic


# Select features
X = data.drop( columns=['traffic_volume', 'traffic_class'] )
y = data['traffic_class']


# Normalize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split( X_scaled, y, test_size=0.2, random_state=42 )


# Train a Decision Tree Classifier
clf = DecisionTreeClassifier( max_depth=4, random_state=42 )
clf.fit( X_train, y_train )

# Plot the Decision Tree
plt.figure( figsize=( 12, 6 ) )
plot_tree( clf, feature_names = X.columns, class_names=[ 'Low Traffic', 'High Traffic' ], filled=True )
plt.title("Decision Tree as a Surrogate Model for Traffic Classification")
plt.show()
