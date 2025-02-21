import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the Dataset
df = pd.read_csv(r'C:\Users\Dell\OneDrive\Desktop\Drugsanalysis\drugsideffects.csv')
#information
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Handling missing values selectively
categorical_columns = ['drug_name', 'medical_condition', 'side_effects', 'generic_name', 
                       'drug_classes', 'pregnancy_category', 'rx_otc', 'csa']
df[categorical_columns] = df[categorical_columns].fillna('Unknown')

numeric_columns = ['rating', 'no_of_reviews', 'activity']
df[numeric_columns] = df[numeric_columns].fillna(0)

# Ensure numeric columns are properly formatted
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
df['no_of_reviews'] = pd.to_numeric(df['no_of_reviews'], errors='coerce')
df['activity'] = pd.to_numeric(df['activity'], errors='coerce')

# Summary statistics
print(df.describe())

# Distribution of Ratings #EDA #graph1
plt.figure(figsize=(10, 6))
sns.histplot(df['rating'], bins=10, kde=True)
plt.title('Distribution of Drug Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.show()

# Top Drugs by Condition
top_drugs = df.groupby('medical_condition')['drug_name'].value_counts().nlargest(10)
print(top_drugs)

# Side Effects Analysis
side_effects = df['side_effects'].value_counts().head(10)
print(side_effects)

# Drug Ratings by Class #graph2
plt.figure(figsize=(12, 8))
sns.boxplot(x='drug_classes', y='rating', data=df)
plt.xticks(rotation=90)
plt.title('Drug Ratings by Class')
plt.show()

# Encoding categorical data #Categorical columns to numeric values
label_encoder = LabelEncoder()
df['generic_name'] = label_encoder.fit_transform(df['generic_name'])
df['medical_condition'] = label_encoder.fit_transform(df['medical_condition'])
df['side_effects'] = label_encoder.fit_transform(df['side_effects'])
df['pregnancy_category'] = label_encoder.fit_transform(df['pregnancy_category'])
df['rx_otc'] = label_encoder.fit_transform(df['rx_otc'])
df['csa'] = label_encoder.fit_transform(df['csa'])

# Feature Scaling
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df.select_dtypes(include=[np.number])), columns=df.select_dtypes(include=[np.number]).columns)

# Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df_scaled.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

# Dropping Duplicates
duplicate_rows = df[df.duplicated()]
print("Count of Duplicate Rows:", duplicate_rows.shape[0])
df = df.drop_duplicates()

# Creating new Columns for Specific Side Effects
def has_hives(text):
    return 'hives' in str(text).lower()
df['Hives'] = df['side_effects'].apply(has_hives)

def has_difficult_breathing(text):
    return 'difficult breathing' in str(text).lower() or 'difficulty breathing' in str(text).lower()
df['Difficult Breathing'] = df['side_effects'].apply(has_difficult_breathing)

def has_itching(text):
    return 'itching' in str(text).lower()
df['Itching'] = df['side_effects'].apply(has_itching)

# Save cleaned dataset
df.to_csv(r'C:\Users\Dell\OneDrive\Desktop\Drugsanalysis\cleaned_drug_data.csv', index=False)

# Plot the count of occurrences for each side effect
plt.figure(figsize=(10, 5))
df['Hives'].value_counts().plot(kind='bar')
plt.title('Count of Hives')
plt.xlabel('Hives')
plt.ylabel('Count')
plt.xticks([0, 1], ['False', 'True'], rotation=0)
plt.show()

plt.figure(figsize=(10, 5))
df['Difficult Breathing'].value_counts().plot(kind='bar')
plt.title('Count of Difficult Breathing')
plt.xlabel('Difficult Breathing')
plt.ylabel('Count')
plt.xticks([0, 1], ['False', 'True'], rotation=0)
plt.show()

plt.figure(figsize=(10, 5))
df['Itching'].value_counts().plot(kind='bar')
plt.title('Count of Itching')
plt.xlabel('Itching')
plt.ylabel('Count')
plt.xticks([0, 1], ['False', 'True'], rotation=0)
plt.show()

# Save the processed data
df.to_csv(r'C:\Users\Dell\OneDrive\Desktop\Drugsanalysis\processed_drug_data.csv', index=False)
