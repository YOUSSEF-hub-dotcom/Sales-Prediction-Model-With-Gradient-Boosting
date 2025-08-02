import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error

# Questions we will answer
# Top 10 sales operation
# Average of sales for every Region
# Average of sales for every Product
# Count of each Product
# Mean of Sales
# Mean of price for each unit of product in each region
# what is the top Product with total Sales
# what is the last Product with total Sales
# which sales method is the most
# which sales method is the most in every Region
# How much product at every year
# what about sales over years
# which Retailer is the most
# Is they any relation between features

df = pd.read_csv("C:\\Users\\sara\\Downloads\\Copy of AdidasUSSales_a1_kit102adeed.csv")
pd.set_option('display.width',None)
print(df.head(20))
print("-------------------------------")
print("===========>>> Basics Function:")
print("Information about Data:")
print(df.info())
print("number of rows and columns:")
print(df.shape)
print("The name of columns:")
print(df.columns)
print("Statistical Operation:")
print(df.describe().round())
print("Data types in Dataset:")
print(df.dtypes)
print("Display the index range:")
print(df.index)
print("number of frequency rows:")
print(df.duplicated().sum())

print("-------------------------------")
print("===========>>> Cleaning Data:")
missing_values = df.isnull().mean() * 100
print("The Percentage of missing values in data:\n",missing_values)
print("Missing Values Before Cleaning:")
print(df.isnull().sum())
print("The missing values in data didn't exceed 1% so we deal with by fillna")
df['Retailer'] = df['Retailer'].fillna(df['Retailer'].mode()[0])
df['Retailer_ID'] = df['Retailer_ID'].fillna(df['Retailer_ID'].mean())
df['Invoice_Date'] = df['Invoice_Date'].fillna(df['Invoice_Date'].mode()[0])
df['Region'] = df['Region'].fillna(df['Region'].mode()[0])
df['State'] = df['State'].fillna(df['State'].mode()[0])
df['City'] = df['City'].fillna(df['City'].mode()[0])
df['Product'] = df['Product'].fillna(df['Product'].mode()[0])
df['Price_per_Unit'] = df['Price_per_Unit'].fillna(df['Price_per_Unit'].mean())
df['Units_Sold'] = df['Units_Sold'].fillna(df['Units_Sold'].mean())
df['Total_Sales'] = df['Total_Sales'].fillna(df['Total_Sales'].mean())
df['Operating_Profit'] = df['Operating_Profit'].fillna(df['Operating_Profit'].mean())
df['Operating_Margin'] = df['Operating_Margin'].fillna(df['Operating_Margin'].mean())
df['Sales_Method'] = df['Sales_Method'].fillna(df['Sales_Method'].mode()[0])
print("Is There any columns contain miss values?")
print(df.isnull().sum())
sns.heatmap(df.isnull())
plt.title('The Dataset After Cleaning')
plt.show()

print("-------------------------------")
print("===========>>> Exploration Data Analysis(EDA):")
"""
['Retailer', 'Retailer_ID', 'Invoice_Date', 'Region', 'State', 'City',
       'Product', 'Price_per_Unit', 'Units_Sold', 'Total_Sales',
       'Operating_Profit', 'Operating_Margin', 'Sales_Method']
"""
print("convert Invoice_Date column to datetime")
df['Invoice_Date'] = pd.to_datetime(df['Invoice_Date'],format='%d/%m/%Y')
print("Invoice_Date:\n",df['Invoice_Date'])
df['month'] = df['Invoice_Date'].dt.month
df['year'] = df['Invoice_Date'].dt.year
print(df)
#---------------------------------------
print("Text Data")
print("select region column")
print(df['Region'])
print("Statistical operation")
print(df.describe(include='object').round())
print("what are the unique values in data(object or text)")
#---------------------------------------
for col in df.describe(include='object').round():
    print(col)
    print(df[col].unique())
    print('-'*75)
print("what are the maximum values in data(object or text)")
#---------------------------------------
for col in df.describe(include='object').round():
    print(col)
    print(df[col].str.len().max())
    print('-'*75)
#---------------------------------------
print("How many numbers of unique values in data(object or text)")
#---------------------------------------
for col in df.describe(include='object').round():
    print(col)
    print(df[col].nunique())
    print('-'*75)
print("what are the number of Frequency values in data(object or text)")
#---------------------------------------
for col in df.describe(include='object').round():
    print(col)
    print(df[col].value_counts())
    print('-'*75)
print("what are the most frequent values in data(object or text)")
#---------------------------------------
for col in df.describe(include='object').round():
    print(col)
    print(df[col].mode()[0])
    print('-'*75)
print("The number of non-empty values in data(object or text)")
#---------------------------------------
for col in df.describe(include='object').round():
    print(col)
    print(df[col].count())
    print('-'*75)
#---------------------------------------
print("Numeric Data")
print("The unique values in Price_per_Unit column")
print(df['Price_per_Unit'].unique)
print("The number unique values in Price_per_Unit column")
print(df['Price_per_Unit'].nunique)
print("The most Top 10 Sales in Data")
print(df['Total_Sales'].value_counts().head(10))
print("The least  10 Sales in Data")
print(df['Total_Sales'].value_counts().tail(10))

print("Average of sales for every Region")
Avg_Sal_Reg = df.groupby('Region')['Total_Sales'].mean()
print(Avg_Sal_Reg)

print("Aggregation Function to sales for every Region")
Agg_Sal_Reg = df.groupby('Region')['Total_Sales'].agg(['sum','mean','max','min','count'])
print(Agg_Sal_Reg)

print("Average of sales for every Product")
Avg_Sal_Pro = df.groupby('Product')['Total_Sales'].mean()
print(Avg_Sal_Pro)

print("Aggregation Function to sales for every Product")
Agg_Sal_Pro = df.groupby('Product')['Total_Sales'].agg(['sum','mean','max','min','count'])
print(Agg_Sal_Pro)

print("Count of each Product")
print(df['Product'].count())
print("Average of Sales")
print(df['Total_Sales'].mean())

print("Mean of price for each unit of product in each region")
print(df.groupby(['Region','Product'])['Price_per_Unit'].agg(['mean']))
print("which sales method is the most")
print(df.groupby('Sales_Method')['Total_Sales'].agg(['sum','mean','min','max']))
print("What is the Top 10 sales operation")
print(df.sort_values(by=['Total_Sales'],ascending=False).head(10))
print("Frequency values in Sales_Method column")
sales_method = df['Sales_Method'].value_counts()
print(sales_method)
print("Most common Sales Method in every Region:")
most_common_sales_method = df.groupby("Region")['Sales_Method'].agg(lambda x: x.value_counts().idxmax)
print(most_common_sales_method)
print("Number of unique products in every year:")
product_count_per_year = df.groupby('year')['Product'].nunique()
print(product_count_per_year)
print("Total Sales Over the years:")
sales_over_years = df.groupby('year')['Total_Sales'].sum()
print(sales_over_years)
print("Most Frequent Retailer")
most_frequent_retailer = df['Retailer'].value_counts().idxmax
print(most_frequent_retailer)
print("Correlation Matrix")
numerical_cols = ['Price_per_Unit','Units_Sold','Total_Sales','Operating_Profit','Operating_Margin']
correlation_matrix = df[numerical_cols].corr()
print(correlation_matrix)
print("Average Profits by Region")
Avg_profit_region = df.groupby("Region")['Operating_Profit'].mean()
print(Avg_profit_region)
print("-------------------------------")
print("===========>>> Visualization:")
plt.figure(figsize=(10,6))
Avg_Sal_Reg.plot(kind='bar',color='skyblue')
plt.title('Average Sales by Region',fontsize=14,fontweight='bold',pad=15)
plt.xlabel("Region",fontsize=12)
plt.ylabel("Average Sales",fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y',linestyle='--',alpha=0.7)
plt.tight_layout()
plt.show()
#--------------------------------------------
plt.figure(figsize=(10,6))
Avg_Sal_Pro.plot(kind='bar',color='lightgreen')
plt.title('Average Sales by Product',fontsize=14,fontweight='bold',pad=15)
plt.xlabel("Product",fontsize=12)
plt.ylabel("Average Sales",fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y',linestyle='--',alpha=0.7)
plt.tight_layout()
plt.show()
#---------------------------------------------
plt.figure(figsize=(12,8))
sales_method = df['Sales_Method'].value_counts()
plt.title("Frequency values in Sales_Method",fontsize=14)
plt.pie(sales_method,labels=sales_method.index,autopct='%1.1f%%')
plt.show()
#---------------------------------------------
plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix,annot=True,cmap='coolwarm',vmin=-1,vmax=1)
plt.title('Correlation Between Numerical Features',fontsize=14)
plt.show()
#---------------------------------------------
plt.figure(figsize=(10,6))
sales_over_years.plot(kind='line',marker='o',color='blue')
plt.title("Total Sales Over Years",fontsize=14,fontweight='bold',pad=15)
plt.xlabel("Year",fontsize=12)
plt.ylabel("Total Sales",fontsize=12)
plt.grid(axis='y',alpha=0.7)
plt.tight_layout()
plt.show()
#---------------------------------------------
plt.figure(figsize=(10,8))
Avg_profit_region.plot(kind='bar',color='salmon',edgecolor='black')
plt.title("Average Profit by Region",fontsize=14,fontweight='bold',pad=15)
plt.xlabel("Region",fontsize=12)
plt.ylabel("Average Profit",fontsize=12)
plt.grid(axis='y',alpha=0.7)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
#---------------------------------------------
sales_over_time = df.groupby(['year', 'month'])['Total_Sales'].sum().reset_index()
plt.figure(figsize=(12, 6))
sns.lineplot(data=sales_over_time, x='month', y='Total_Sales', hue='year', marker='o', palette='tab10')
plt.title('Total Sales Over Months by Year', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Month', fontsize=12)
plt.ylabel('Total Sales', fontsize=12)
plt.grid(axis='y', alpha=0.7)
plt.legend(title='Year')
plt.tight_layout()
plt.show()
#---------------------------------------------

print("----------------------- Machine Learning ------------------")
"""
['Retailer', 'Retailer_ID', 'year','month', 'Region', 'State', 'City',
       'Product', 'Price_per_Unit', 'Units_Sold', 'Total_Sales','year','month'
       'Operating_Profit', 'Operating_Margin', 'Sales_Method']
"""
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load and clean data
df = df.dropna()

# Remove outliers using IQR
Q1 = df['Total_Sales'].quantile(0.25)
Q3 = df['Total_Sales'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['Total_Sales'] >= Q1 - 1.5 * IQR) & (df['Total_Sales'] <= Q3 + 1.5 * IQR)]

# Prepare the data
X = df[['Price_per_Unit', 'Units_Sold', 'Operating_Profit', 'Operating_Margin', 'Region', 'Product', 'Sales_Method']]
y = df['Total_Sales']

# Handle categorical variables
categorical_cols = ['Region', 'Product', 'Sales_Method']
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_cols = encoder.fit_transform(df[categorical_cols])
encoded_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out(categorical_cols))

# Combine and scale numerical columns
X_numeric = df[['Price_per_Unit', 'Units_Sold', 'Operating_Profit', 'Operating_Margin']]
scaler = StandardScaler()
X_numeric_scaled = scaler.fit_transform(X_numeric)
X = pd.concat([pd.DataFrame(X_numeric_scaled, columns=X_numeric.columns), encoded_df], axis=1)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)

# Hyperparameter tuning for GradientBoostingRegressor
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10],
    'learning_rate': [0.01, 0.1]
}
gb_model = GradientBoostingRegressor(random_state=42)
grid_search = GridSearchCV(gb_model, param_grid, cv=5, scoring='r2')
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")

# Train the best model
best_model.fit(X_train, y_train)

# Predict and evaluate
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R-squared Score: {r2:.2f}")

# Cross-validation
cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='r2')
print(f"Cross-validated R² Score: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")