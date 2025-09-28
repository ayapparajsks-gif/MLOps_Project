#1. Data Processing & Ingestion

import pandas as pd

#Loading the kaggle shopping mall dataset
df=pd.read_csv("/home/vboxuser/Downloads/customer_shopping_data.csv")

#Changing the format for invoice_date column
df['invoice_date']=pd.to_datetime(df['invoice_date'],format='%d/%m/%Y')

print(df.head())

print(df.info())

#Checking for missing values
print(df.isnull().sum())

# Store vs. Region Performance

import matplotlib.pyplot as plt
import seaborn as sns

# Adding revenue calculation column to the dataframe
df['revenue']=df['quantity']*df['price']

print(df.head())

#Shopping mall wise revenue to analyze store performance
store_performance=df.groupby('shopping_mall')['revenue'].sum().sort_values(ascending=False)
store_performance

# Vizualising the store performance
plt.figure(figsize=(12,6))
sns.barplot(x=store_performance.index,y=store_performance.values)
plt.xticks(rotation=45,ha='right')
plt.title('Total revenue by shopping maill')
plt.xlabel('Shopping mall')
plt.ylabel('Total revenue')

print(store_performance)

# Top Customers - Identifying top 10% of customers based on their total purchase value

customer_spending=df.groupby('customer_id')['revenue'].sum().reset_index()

top_10_percent_threshold=customer_spending['revenue'].quantile(0.9)

top_customers=customer_spending[customer_spending['revenue']>=top_10_percent_threshold]

print("Top 10% customers (by purchase value):")
print(top_customers)

# 3. High vs. Low value segmentation - Based on total spending

median_spending = customer_spending['revenue'].median()
customer_spending['segment']=['High-value' if x>=median_spending else 'Low-value' for x in customer_spending['revenue']]

print("\nCustomer segmentation (High vs. Low value):")
print(customer_spending.head())

# Visualizing segmentation

import matplotlib.pyplot as plt
import seaborn as sns

segment_counts=customer_spending['segment'].value_counts()

plt.figure(figsize=(6,6))

sns.barplot(x=segment_counts.index,y=segment_counts.values)

plt.title('High vs. Low value customer segmentation')
plt.xlabel('Customer segment')
plt.ylabel('Number of customers')

plt.show()

print(df.head())

# 4. Adding margin column based on 10% discount

discounted_percentage=0.10

df['discount']=df['price']*discounted_percentage

df['margin']=df['price']-df['discount']

df['revenue_margin']=df['margin']*df['quantity']

print(df.head())

# 5. Seasonal trend analysis

df['year_month']=df['invoice_date'].dt.to_period('M')

monthly_sales=df.groupby('year_month')['revenue'].sum().reset_index()
monthly_sales['year_month']=monthly_sales['year_month'].dt.to_timestamp()

df.head()

#Vizualising seasonal analysis

plt.figure(figsize=(12,6))

plt.plot(monthly_sales['year_month'],monthly_sales['revenue'],marker='o')

plt.title('Monthly sales trend')
plt.xlabel('Month')
plt.ylabel('Total revenue')

plt.grid(True)
plt.show()

# 5. Seasonal trend analysis - Quarterly

df['quarter_month']=df['invoice_date'].dt.to_period('Q')

monthly_sales=df.groupby('quarter_month')['revenue'].sum().reset_index()
monthly_sales['quarter_month']=monthly_sales['quarter_month'].dt.to_timestamp()

df.head()

#Vizualising seasonal analysis

plt.figure(figsize=(12,6))

plt.plot(monthly_sales['quarter_month'],monthly_sales['revenue'],marker='o')

plt.title('Quarterly sales trend')
plt.xlabel('Quarter')
plt.ylabel('Total revenue')

plt.grid(True)
plt.show()

# 6. Payment method preference

payment_method_counts=df['payment_method'].value_counts()

plt.figure(figsize=(8,8))

plt.pie(payment_method_counts,labels=payment_method_counts.index,autopct='%1.1f%%',startangle=140)
plt.title('Payment method preference')
plt.axis('equal')
plt.show()

print(payment_method_counts)

# 7. Customer segmentation - RFM analysis to segment customers

#Recency
max_date=df['invoice_date'].max()
df['recency']=(max_date-df['invoice_date']).dt.days
print(df.head())

#Frequency
frequency_df=df.groupby('customer_id')['invoice_no'].count().reset_index()
frequency_df.rename(columns={'invoice_no':'frequency'},inplace=True)
frequency_df

#Monetary
monetary_df=df.groupby('customer_id')['revenue'].sum().reset_index()
monetary_df.rename(columns={'invoice_no':'frequency'},inplace=True)
monetary_df

#COmbining Recency, Frequency and Monetary dataframes

rfm_df = df.groupby('customer_id').agg(
    {
     'recency':'min',
     'invoice_no':'count',
     'revenue':'sum'        
    }).reset_index()

rfm_df.rename(columns=
    {
        'invoice_no':'frequency',
        'revenue':'monetary'
    }, inplace=True)

rfm_df

#Segmentation

rfm_df['r_score']=pd.qcut(rfm_df['recency'],4,labels=[4,3,2,1])
rfm_df['f_score']=pd.qcut(rfm_df['frequency'].rank(method='first'),4,labels=[1,2,3,4])
rfm_df['m_score']=pd.qcut(rfm_df['monetary'],4,labels=[1,2,3,4])
rfm_df['rfm_segment']=rfm_df['r_score'].astype(str)+rfm_df['f_score'].astype(str)+rfm_df['m_score'].astype(str)

rfm_df


#Customer segmentation definition

def cust_seg(rfm_score):
    if rfm_score in ['444','443','434','344']:
        return 'Best Customers'
    elif rfm_score in ['333','334','343','433']:
        return 'Loyal Customers'    
    elif rfm_score in ['441','442','431','432','421','422','341','342','331','332']:
        return 'Potential Loyalists'
    elif rfm_score in ['111','112','121','122','211','212','221','222']:
        return 'Lost Customers'
    else:
        return 'Regular Customers'


rfm_df['customer_segment']=rfm_df['rfm_segment'].apply(cust_seg)
print(rfm_df.head())

#Vizualising customer segment

plt.figure(figsize=(10,6))
sns.countplot(y='customer_segment',data=rfm_df,order=rfm_df['customer_segment'].value_counts().index)

plt.title('Customer segmentation based on RFM analysis')
plt.xlabel('Number of customers')
plt.ylabel('Customer Segment')

plt.show()

# 8. Repeat customer vs One-time - Sales contribution

customer_frequency=df.groupby('customer_id')['invoice_no'].count().reset_index()

customer_frequency['customer_type']=['Repeat' if x>1 else 'One-time' for x in customer_frequency['invoice_no']]

merged_df=pd.merge(df,customer_frequency,on='customer_id')
sales_contribution=merged_df.groupby('customer_type')['revenue'].sum()

print("\n Sales contribution (Repeat vs One-time customers):")
print(sales_contribution)

plt.figure(figsize=(6,6))
sales_contribution.plot(kind='pie',autopct='%1.1f%%',startangle=90)

plt.title('Sales contribution by customer type')
plt.ylabel('')
plt.show()

# 9. Category wise insights - identify most profitable product categories and customer segments that purchase from the mall

category_profit=df.groupby('category')['revenue'].sum().sort_values(ascending=False)

print("\nMost valuable categories:")
print(category_profit)

plt.figure(figsize=(12,6))

sns.barplot(x=category_profit.index,y=category_profit.values)

plt.xticks(rotation=45,ha='right')

plt.title('Total revenue by product category')
plt.xlabel('Product category')
plt.ylabel('Total revenue')

plt.show()

# 10. Campaign simulation - ROI of 10% discount campaign targeting high value customers

high_value_threshold=customer_spending['revenue'].quantile(0.75)
high_value_customers=customer_spending[customer_spending['revenue']>=high_value_threshold]

# Assuming 20% increase in purchase frequency for high value customers due to 10% discount

simulated_revenue_increase=high_value_customers['revenue'].sum()*0.20
cost_of_discount=high_value_customers['revenue'].sum()*0.10

projected_roi=(simulated_revenue_increase-cost_of_discount)/cost_of_discount

print(f"\nCampaign simulation (10% discount for high value customers):")

print(f"Projected revenue increase: ${simulated_revenue_increase:,.2f}")

print(f"Cost of discount: ${cost_of_discount:,.2f}")

print(f"Projected ROI: {projected_roi:.2%}")

#Predictive Model - Clustering customers based on their RFM scores using K-means

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

X=rfm_df[['recency','frequency','monetary']]

scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)

wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10,5))

plt.plot(range(1,11),wcss,marker='o')

plt.title('Elbow method for optimal number of clusters')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')

plt.show()

kmeans=KMeans(n_clusters=4,init='k-means++',max_iter=300,n_init=10,random_state=0)

clusters=kmeans.fit_predict(X_scaled)
rfm_df['cluster']=clusters

print(rfm_df.head())

rfm_df.to_csv('rfm_data.csv',index=False)

print(rfm_df.head())
