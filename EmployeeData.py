import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv('employeerecord.csv')

# Drop irrelevant columns and rows with missing values
df = df.drop(columns=['Unnamed: 25', 'Unnamed: 26', 'Unnamed: 27']).dropna()

# Convert all month columns to float
month_columns = df.columns[3:]  # All columns from Jun-22 onwards
df[month_columns] = df[month_columns].astype(float)

# Display the first few rows of the cleaned dataset
print("First few rows of the cleaned dataset:")
print(df.head())

# Basic data exploration
# Distribution of average scores
df['AverageScore'] = df[month_columns].mean(axis=1)

plt.figure(figsize=(10, 6))
sns.histplot(df['AverageScore'], kde=True)
plt.title('Distribution of Average Scores')
plt.xlabel('Average Score')
plt.ylabel('Frequency')
plt.show()

# Clustering employees based on average scores and performance in the last month
df['LastMonthScore'] = df['Mar-24']  # Assuming Mar-24 is the last month
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(df[['AverageScore', 'LastMonthScore']])

# Visualize the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x='AverageScore', y='LastMonthScore', data=df, hue='Cluster', palette='viridis')
plt.title('Employee Clusters')
plt.xlabel('Average Score')
plt.ylabel('Last Month Score')
plt.legend(title='Cluster')
plt.show()

# Predicting the last month's performance using Linear Regression
X = df[['AverageScore']]
y = df['LastMonthScore']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Make predictions
y_pred = regressor.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nMean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Visualize the regression results
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Last Month Scores')
plt.show()