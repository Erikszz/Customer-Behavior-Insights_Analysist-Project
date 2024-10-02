import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import streamlit as st

sns.set(style='dark')
df = pd.read_csv('dashboard/main.csv', parse_dates=['order_purchase_timestamp'])


def create_monthly_orders_df(df):
    monthly_orders_df = df.resample(rule='M', on='order_purchase_timestamp').agg({
        "order_id": "count",
        "price": "sum"
    })
    monthly_orders_df.index = monthly_orders_df.index.strftime('%Y-%m')
    monthly_orders_df = monthly_orders_df.reset_index()

    monthly_orders_df.rename(columns={
        "order_id": "order_count",
        "price": "revenue",
        "order_purchase_timestamp": "order_date"
    }, inplace=True)

    # the last transaction on this month is on 3rd September
    monthly_orders_df = monthly_orders_df[monthly_orders_df['order_date'] != '2018-09']
    
    return monthly_orders_df

def create_rfm_df(df):
    rfm_df = df.groupby(by="customer_id", as_index=False).agg({
        "order_purchase_timestamp": "max",
        "order_id": "count",
        "price": "sum"
    })

    rfm_df.columns = ["customer_id", "max_order_timestamp", "frequency", "monetary"]
    rfm_df["max_order_timestamp"] = rfm_df["max_order_timestamp"].dt.date
    recent_date = df["order_purchase_timestamp"].dt.date.max()
    rfm_df["recency"] = rfm_df["max_order_timestamp"].apply(lambda x: (recent_date - x).days)
    rfm_df.drop("max_order_timestamp", axis=1, inplace=True)

    return rfm_df

def create_rfm_analysis(rfm_df):
    # rfm scores
    rfm_df['R_score'] = rfm_df['recency'].apply(lambda x: 3 if x <= 30 else (2 if x <= 90 else 1))
    rfm_df['F_score'] = rfm_df['frequency'].apply(lambda x: 3 if x >= 15 else (2 if x >= 7 else 1))
    rfm_df['M_score'] = pd.qcut(rfm_df['monetary'], q=[0, 0.4, 0.8, 1], labels=[1, 2, 3])

    # segmentation
    rfm_df['value_segment'] = pd.qcut(rfm_df['monetary'], q=[0, 0.4, 0.8, 1], labels=['Low Value', 'High Value', 'Special Value'])
    rfm_df['recency_segment'] = rfm_df['recency'].apply(lambda x: 'Active' if x <= 30 else ('Warm' if x <= 90 else 'Cold' if x <= 180 else 'Inactive'))

    monetary_sum_value = rfm_df.groupby(by='value_segment')['monetary'].sum()
    monetary_sum_recency = rfm_df.groupby(by='recency_segment')['monetary'].sum()
    monetary_percent_value = (monetary_sum_value / monetary_sum_value.sum()) * 100
    monetary_percent_recency = (monetary_sum_recency / monetary_sum_recency.sum()) * 100

    return monetary_percent_value, monetary_percent_recency

def create_heatmap_rfm(rfm_df):
    pivot_table = rfm_df.pivot_table(index='value_segment', columns='recency_segment', aggfunc='size', fill_value=0)
    pivot_table_percent = (pivot_table / pivot_table.sum().sum()) * 100

    return pivot_table_percent

def create_avg_monetary_transaction_per_cust(rfm_df):
    total_transaction = rfm_df['monetary'].sum()
    avg_transaction = round(total_transaction / len(rfm_df), 2)
    return avg_transaction

monthly_orders = create_monthly_orders_df(df)
rfm_df = create_rfm_df(df)
monetary_percent_value, monetary_percent_recency = create_rfm_analysis(rfm_df)
pivot_table_percent = create_heatmap_rfm(rfm_df)

st.header('RFM Analysis of E-Commerce')
st.subheader('Insights into Customer Segmentation and Revenue Based on RFM Analysis')
col1, col2, col3 = st.columns(3)

with col1:
    total_orders = df['order_id'].count()
    st.metric('Total Orders', value=total_orders)

with col2:
    revenue_based_monetary = round(rfm_df['monetary'].sum(), 2)
    st.metric('Revenue Based on Monetary', value=revenue_based_monetary)

with col3:
    avg_monetary_transaction_per_cust = create_avg_monetary_transaction_per_cust(rfm_df)
    st.metric('Avg Transaction per Cust', value=avg_monetary_transaction_per_cust)

# create bar plot
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(30, 6))
colors = ["#72BCD4"] * 5

sns.barplot(y="recency", x="customer_id", data=rfm_df.sort_values(by="recency", ascending=True).head(5), palette=colors, ax=ax[0])
ax[0].set_title("By Recency (days)", fontsize=18)
ax[0].set_xlabel(None) 
ax[0].set_ylabel(None)
ax[0].tick_params(axis='x', rotation=70)

sns.barplot(y="frequency", x="customer_id", data=rfm_df.sort_values(by="frequency", ascending=False).head(5), palette=colors, ax=ax[1])
ax[1].set_title("By Frequency", fontsize=18)
ax[1].set_xlabel(None) 
ax[1].set_ylabel(None)
ax[1].tick_params(axis='x', rotation=70)

sns.barplot(y="monetary", x="customer_id", data=rfm_df.sort_values(by="monetary", ascending=False).head(5), palette=colors, ax=ax[2])
ax[2].set_title("By Monetary", fontsize=18)
ax[2].set_xlabel(None) 
ax[2].set_ylabel(None)
ax[2].tick_params(axis='x', rotation=70)
plt.suptitle("Best Customer Based on RFM Parameters (customer_id)", fontsize=22)
st.pyplot(plt)

st.markdown("""
The RFM analysis provides valuable insights into customer behavior that can inform strategic business decisions. Based on recency, the data reveals that the time gap between a customer's last purchase and their next order is minimal. Notably, the first segment comprises customers who make repeat purchases within the same day, followed by four additional customers who also return after just five days. This pattern indicates a high level of customer loyalty, suggesting that the e-commerce platform successfully cultivates a loyal customer base eager to shop again shortly after their last transaction.

When examining frequency, it becomes apparent that some customers exhibit remarkable purchasing behavior, with one individual placing up to 60 repeat orders. This is complemented by additional customers who exceed 20 orders, highlighting not only customer satisfaction with the products and services offered but also the potential for increased revenue through repeat business. Such patterns indicate that the e-commerce platform meets customer needs effectively, creating a reliable shopping experience that encourages loyalty and ongoing patronage.

From a monetary perspective, the analysis shows customers making substantial transactions, with the highest total reaching 4.000, followed closely by others exceeding 2.000. This suggests that customers perceive significant value in the offerings, further affirming their satisfaction with both the products and services provided. Understanding these dynamics enables the e-commerce business to tailor marketing efforts, enhance customer engagement strategies, and ultimately drive sustained growth by focusing on retaining and nurturing their most valuable customers.
""")


# create plot for revenue
plt.figure(figsize=(10, 6))
plt.plot(monthly_orders['order_date'], monthly_orders['revenue'], marker='o', color='lightblue')
plt.title('Monthly Revenue')
plt.tick_params(axis='x', rotation=45) 
plt.xticks(monthly_orders['order_date'][::3], rotation=45)
plt.grid(False)
plt.tight_layout()
st.title('Monthly Revenue in 3 Years')
st.pyplot(plt)
st.markdown("""
The visualizations reveal a significant growth revenue patterns over time. The monthly revenue shows a notable increase, particularly between September 2017 and before December 2017, revenue peaking at nearly 700.000 during this period. This surge reflects the effectiveness of marketing strategies and customer engagement initiatives, which have successfully driven sales growth. However, there is a significant decline in revenue, dropping back to almost 500.000, before stabilizing to constant growth. This downward trend suggests potential operational challenges that may need to be addressed to maintain customer satisfaction and retention.

Despite the downturn in revenue, the data shows a strong growth in order count, indicating the company’s resilience and ability to effectively engage customers. This suggests that while customers are still making purchases, there may be issues affecting their overall spending behavior, which could be linked to factors such as product availability, customer experience, or market competition. By analyzing these trends, businesses can identify the underlying causes of the revenue decline and take proactive measures to improve performance.

Overall, these insights underscore the necessity for continuous monitoring of revenue metrics to enhance business strategies. By closely analyzing the growth patterns of revenue, the e-commerce can informed decisions to optimize its marketing efforts, improve customer satisfaction, and adapt to market changes. This proactive approach is essential for sustaining long-term growth and ensuring the platform remains competitive in the evolving e-commerce landscape.
""")

# create color lists: gray for all, color for the max
colors_value = ['lightgray'] * len(monetary_percent_value)
colors_recency = ['lightgray'] * len(monetary_percent_recency)

# highlight the maximum values
max_value_index = np.argmax(monetary_percent_value.values)
colors_value[max_value_index] = 'lightblue'  
max_recency_index = np.argmax(monetary_percent_recency.values)
colors_recency[max_recency_index] = 'lightblue' 


fig, axes = plt.subplots(1, 2, figsize=(14, 6))

bars_value = axes[0].barh(monetary_percent_value.index, monetary_percent_value.values, color=colors_value)
for bar in bars_value:
    xval = bar.get_width()
    axes[0].text(xval, bar.get_y() + bar.get_height()/2, f'{xval:.2f}%', ha='left', va='center', fontsize=12)

ax[0].set_title('Monetary Distribution by Value Segment', fontsize=14)
ax[0].set_xlabel(None, fontsize=12) 
ax[0].set_ylabel(None, fontsize=12)

bars_recency = axes[1].barh(monetary_percent_recency.index, monetary_percent_recency.values, color=colors_recency)
for bar in bars_recency:
    xval = bar.get_width()
    axes[1].text(xval, bar.get_y() + bar.get_height()/2, f'{xval:.2f}%', ha='left', va='center', fontsize=12)

ax[1].set_title('Monetary Distribution by Recency Segment', fontsize=14)
ax[1].set_xlabel(None, fontsize=12) 
ax[1].set_ylabel(None, fontsize=12) 

plt.tight_layout()
plt.show()
st.title('Monetary Distribution')
st.pyplot(plt)

st.markdown("""
The monetary distribution by value segment illustrates a strong financial performance for the e-commerce, with a substantial portion of revenue generated from customers classified as "special value" at 44.19% and "high value" at 41.04%. In contrast, customers in the "low value" segment account for only 14.77% of total revenue. This distribution indicates that a significant majority of customers possess high purchasing power, which is crucial for driving the growth and sustainability of the e-commerce business. By focusing on retaining and engaging these high-value customers, the platform can further enhance its revenue streams and ensure long-term profitability.

Conversely, the monetary distribution by recency segment reveals concerning trends, with a staggering 61.12% of customers categorized as inactive, followed by 20.63% as cold customers. This data suggests that a large number of customers have not made a purchase in over 90 days, raising red flags about customer retention and engagement strategies. It is essential for the business to investigate the factors contributing to this inactivity, whether through understanding customer feedback, enhancing marketing outreach, or improving product offerings. Addressing these issues will be vital for re-engaging these dormant customers, thereby unlocking potential revenue and ensuring a more balanced customer base for sustained growth in the competitive e-commerce landscape. 
""")

# make heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_table_percent, annot=True, fmt='.2f', cmap='Greens', cbar_kws={'format': '%.0f%%'})

plt.xlabel(None, fontsize=12)
plt.ylabel(None, fontsize=12)
st.title('Heatmap of Recency Segment and Value Segment')
st.pyplot(plt)
st.markdown("""
The visualization reveals a notable correlation between the value segment and the recency segment within the e-commerce. A significant portion of inactive customers, particularly those classified in the "low value" segment, demonstrates a concerning trend, with a correlation of 24.95%. This indicates that many customers fail to make repeat purchases after their last transaction, resulting in low transaction amounts that contribute only 0 to 40% of the total revenue percentile. This scenario is problematic as it suggests that while the e-commerce may attract a considerable number of customers, a significant portion of them is not engaging in repeat purchases, which is crucial for sustaining revenue growth.

Furthermore, the second-highest correlation, at 24.31%, not so much differences or less than 1% with the previous correlation (24.95%) indicates that customers who have not made purchases in over 90 days exhibit high purchasing power. This suggests that although these customers have the potential to contribute significantly to revenue, they remain inactive for extended periods. Additionally, there are also inactive customers classified as "special value" with 11.86% as the number of correlation. These findings highlight a disconnect in the customer journey that could be attributed to inadequate marketing strategies or ineffective customer engagement initiatives. Addressing this gap is essential for enhancing customer retention and driving repeat purchases, which are fundamental for the long-term growth of the e-commerce business.

To leverage the potential of these high-value yet inactive customers, the e-commerce platform must develop targeted marketing strategies aimed at re-engaging this segment. Implementing personalized communication, loyalty programs, and special promotions tailored to their preferences could entice these customers back to the platform. Furthermore, the business should conduct a thorough analysis of customer feedback to identify pain points in their shopping experience. By improving marketing strategies and focusing on customer retention, the e-commerce can transform these inactive customers into loyal patrons, ultimately driving growth in both orders and revenue while maximizing the overall customer lifetime value.
""")