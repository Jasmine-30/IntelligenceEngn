import pandas as pd

# Load the uploaded dataset
file_path = 'time_series_data_human_activities.csv'
df = pd.read_csv(file_path)


# Display basic information about the dataset
df_info = df.info()
df_head = df.head()
df_summary = df.describe(include='all')

print(df_info)
print(df_head)
print(df_summary)
print(df.columns)
print(df['activity'].value_counts())

# Drop missing values and duplicates
data_cleaned = df.dropna()
data_cleaned = data_cleaned.drop_duplicates()

from sklearn.preprocessing import StandardScaler

# Standardize numerical features
scaler = StandardScaler()
numeric_features = data_cleaned.select_dtypes(include=['float64', 'int64']).columns
data_cleaned[numeric_features] = scaler.fit_transform(data_cleaned[numeric_features])

X = data_cleaned.drop(['activity'], axis=1)  # Replace 'activity' with the target column name
y = data_cleaned['activity']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)


from sklearn.metrics import accuracy_score, classification_report

y_pred = rf_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Classification report
print(classification_report(y_test, y_pred))


import seaborn as sns # type: ignore
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


plt.figure(figsize=(8, 6))
sns.countplot(x='activity', data=df)
plt.title('Activity Distribution')
plt.xlabel('Activity')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(15, 12))
data_cleaned[numeric_features].hist(bins=20, figsize=(15, 12))
plt.suptitle('Distribution of Numerical Features')
plt.show()


