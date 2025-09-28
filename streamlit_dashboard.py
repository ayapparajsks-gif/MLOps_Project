
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as pl
import seaborn as sns

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Retail Insights Dashboard",
    page_icon="🛒",
    layout="wide",
)

# --- DATA LOADING AND CACHING ---
@st.cache_data
def load_data():
    """Loads, cleans, and prepares the customer shopping data."""
    df = pd.read_csv('customer_shopping_data.csv')
    df['invoice_date'] = pd.to_datetime(df['invoice_date'], format='%d/%m/%Y')
    df['revenue'] = df['quantity'] * df['price']
    df['year_month'] = df['invoice_date'].dt.to_period('M')
    return df

df_original = load_data()

# --- RFM ANALYSIS FUNCTION ---
@st.cache_data
def calculate_rfm(df):
    """Performs RFM analysis and customer segmentation."""
    max_date = df['invoice_date'].max()
    rfm_df = df.groupby('customer_id').agg({
        'invoice_date': lambda date: (max_date - date.max()).days,
        'invoice_no': 'count',
        'revenue': 'sum'
    }).reset_index()

    rfm_df.rename(columns={
        'invoice_date': 'recency',
        'invoice_no': 'frequency',
        'revenue': 'monetary'
    }, inplace=True)

    # Calculate RFM scores
    rfm_df['r_score'] = pd.qcut(rfm_df['recency'], 4, labels=[4, 3, 2, 1])
    rfm_df['f_score'] = pd.qcut(rfm_df['frequency'].rank(method='first'), 4, labels=[1, 2, 3, 4])
    rfm_df['m_score'] = pd.qcut(rfm_df['monetary'], 4, labels=[1, 2, 3, 4])

    # Define customer segments
    def segment_customer(row):
        if row['r_score'] == 4 and row['f_score'] == 4:
            return 'Best Customers'
        elif row['f_score'] >= 3:
            return 'Loyal Customers'
        elif row['r_score'] >= 3:
            return 'Potential Loyalists'
        elif row['r_score'] == 1:
            return 'Lost Customers'
        else:
            return 'Regular Customers'

    rfm_df['customer_segment'] = rfm_df.apply(segment_customer, axis=1)
    return rfm_df


# --- SIDEBAR FILTERS ---
st.sidebar.title("Dashboard Controls")
st.sidebar.markdown("Use the filters below to customize the dashboard view.")

selected_mall = st.sidebar.selectbox(
    "Filter by Shopping Mall",
    options=['All'] + sorted(df_original['shopping_mall'].unique())
)

# Apply filter
if selected_mall == 'All':
    df_filtered = df_original.copy()
else:
    df_filtered = df_original[df_original['shopping_mall'] == selected_mall]

rfm_data = calculate_rfm(df_filtered)

# --- DASHBOARD TITLE ---
st.title(f"🛒 Retail Insights: {selected_mall}")
st.markdown("This dashboard provides a comprehensive analysis of customer shopping behavior.")
st.markdown("---")

# --- SCENARIO TABS ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Performance & Sales", "👥 Customer Segmentation", "📊 Category & Payment",
    "💹 Profitability & Campaigns", "🤖 Predictive Modeling"
])


# --- TAB 1: PERFORMANCE & SALES ---
with tab1:
    st.header("Scenario 1: Store vs. Region Performance")
    store_performance = df_original.groupby('shopping_mall')['revenue'].sum().sort_values(ascending=False).reset_index()
    fig_store = px.bar(store_performance, x='shopping_mall', y='revenue', title='Total Revenue by Shopping Mall',
                       labels={'revenue': 'Total Revenue ($)', 'shopping_mall': 'Shopping Mall'})
    st.plotly_chart(fig_store, use_container_width=True)

    st.header("Scenario 5: Seasonality Analysis")
    monthly_sales = df_filtered.groupby('year_month')['revenue'].sum().reset_index()
    monthly_sales['year_month'] = monthly_sales['year_month'].dt.to_timestamp()
    fig_season = px.line(monthly_sales, x='year_month', y='revenue', title='Monthly Sales Trends', markers=True)
    st.plotly_chart(fig_season, use_container_width=True)


# --- TAB 2: CUSTOMER SEGMENTATION ---
with tab2:
    st.header("Scenario 7: RFM Analysis & Segmentation")
    st.write("Customers are segmented based on their Recency, Frequency, and Monetary scores.")
    segment_dist = rfm_data['customer_segment'].value_counts().reset_index()
    fig_rfm = px.pie(segment_dist, names='customer_segment', values='count', title='Customer Segment Distribution', hole=0.3)
    st.plotly_chart(fig_rfm, use_container_width=True)
    st.dataframe(rfm_data.head())

    st.header("Scenario 2 & 3: Top, High-Value, and Low-Value Customers")
    customer_spending = df_filtered.groupby('customer_id')['revenue'].sum().reset_index()
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top 10% Customers")
        top_10_threshold = customer_spending['revenue'].quantile(0.9)
        top_customers = customer_spending[customer_spending['revenue'] >= top_10_threshold]
        st.dataframe(top_customers)

    with col2:
        st.subheader("High vs. Low-Value Segmentation")
        median_spending = customer_spending['revenue'].median()
        customer_spending['segment'] = ['High-Value' if x >= median_spending else 'Low-Value' for x in customer_spending['revenue']]
        segment_counts = customer_spending['segment'].value_counts()
        st.bar_chart(segment_counts)

    st.header("Scenario 8: Repeat vs. One-Time Customers")
    customer_frequency = df_filtered.groupby('customer_id')['invoice_no'].count().reset_index()
    customer_frequency['customer_type'] = ['Repeat' if x > 1 else 'One-Time' for x in customer_frequency['invoice_no']]
    merged_df = pd.merge(df_filtered, customer_frequency, on='customer_id')
    sales_contribution = merged_df.groupby('customer_type')['revenue'].sum()
    fig_repeat = px.pie(sales_contribution, names=sales_contribution.index, values=sales_contribution.values, title='Sales Contribution by Customer Type')
    st.plotly_chart(fig_repeat, use_container_width=True)

# --- TAB 3: CATEGORY & PAYMENT ---
with tab3:
    st.header("Scenario 9: Category-wise Insights")
    category_profit = df_filtered.groupby('category')['revenue'].sum().sort_values(ascending=False).reset_index()
    fig_cat = px.bar(category_profit, x='category', y='revenue', title='Total Revenue by Product Category')
    st.plotly_chart(fig_cat, use_container_width=True)

    st.header("Scenario 6: Payment Method Preference")
    payment_counts = df_filtered['payment_method'].value_counts()
    fig_payment = px.pie(payment_counts, names=payment_counts.index, values=payment_counts.values, title='Payment Method Distribution')
    st.plotly_chart(fig_payment, use_container_width=True)

# --- TAB 4: PROFITABILITY & CAMPAIGNS ---
with tab4:
    st.header("Scenario 4: Discount Impact on Profitability")
    st.write("This is a hypothetical analysis assuming a discount is applied.")
    discount_rate = st.slider("Select a Discount Rate (%)", 0, 50, 10) / 100.0
    
    df_profit = df_filtered.copy()
    df_profit['discount_amount'] = df_profit['price'] * discount_rate
    df_profit['margin'] = df_profit['price'] - df_profit['discount_amount']
    df_profit['total_margin'] = df_profit['margin'] * df_profit['quantity']
    
    total_margin = df_profit['total_margin'].sum()
    st.metric("Total Profit Margin after Discount", f"${total_margin:,.2f}")
    st.dataframe(df_profit[['category', 'price', 'discount_amount', 'margin', 'quantity', 'total_margin']].head())

    st.header("Scenario 10: Campaign Simulation")
    st.write("This simulation projects the ROI of a 10% discount campaign targeting high-value customers.")
    high_value_threshold = df_original.groupby('customer_id')['revenue'].sum().quantile(0.75)
    high_value_customers_revenue = df_original.groupby('customer_id')['revenue'].sum()
    high_value_customers_revenue = high_value_customers_revenue[high_value_customers_revenue >= high_value_threshold].sum()
    
    simulated_increase_rate = st.slider("Assumed Purchase Increase Rate (%)", 5, 50, 20) / 100.0
    
    simulated_revenue_increase = high_value_customers_revenue * simulated_increase_rate
    cost_of_discount = high_value_customers_revenue * 0.10
    projected_roi = (simulated_revenue_increase - cost_of_discount) / cost_of_discount if cost_of_discount > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Projected Revenue Increase", f"${simulated_revenue_increase:,.2f}")
    col2.metric("Cost of Discount", f"${cost_of_discount:,.2f}")
    col3.metric("Projected ROI", f"{projected_roi:.2%}")

# --- TAB 5: PREDICTIVE MODELING ---
with tab5:
    st.header("Predictive Model: Customer Clustering with K-Means")
    st.write("Customers are clustered into segments based on their RFM values using the K-Means algorithm.")
    
    X = rfm_data[['recency', 'frequency', 'monetary']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Use the Elbow method to find the optimal number of clusters
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)
    
    fig_elbow = px.line(x=range(1, 11), y=wcss, title='Elbow Method for Optimal K', labels={'x': 'Number of Clusters', 'y': 'WCSS'})
    st.plotly_chart(fig_elbow, use_container_width=True)
    
    st.info("Based on the elbow plot, an optimal number of clusters (e.g., 4) can be chosen.")
    
    n_clusters = st.number_input("Select Number of Clusters", 2, 10, 4)
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
    clusters = kmeans.fit_predict(X_scaled)
    rfm_data['cluster'] = clusters
    
    fig_cluster = px.scatter(
        rfm_data,
        x='recency',
        y='monetary',
        color='cluster',
        size='frequency',
        hover_name='customer_id',
        title='Customer Clusters (Recency vs. Monetary)'
    )
    st.plotly_chart(fig_cluster, use_container_width=True)
