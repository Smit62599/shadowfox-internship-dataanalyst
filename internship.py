import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set global aesthetics
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

# -------------------------------------------------------------
# 1. GENERATE SYNTHETIC DATASET (Retail Sales & Operations)
# -------------------------------------------------------------
np.random.seed(42)
n_rows = 1200

dates = pd.date_range(start='2025-01-01', end='2025-12-31', periods=n_rows)
regions = ['North', 'South', 'East', 'West']
categories = {
    'Electronics': ['Laptop', 'Smartphone', 'Headphones', 'Smartwatch'],
    'Apparel': ['Jacket', 'Sneakers', 'Jeans', 'T-Shirt'],
    'Home & Kitchen': ['Blender', 'Coffee Maker', 'Air Fryer', 'Cookware Set'],
    'Beauty & Care': ['Skincare Set', 'Perfume', 'Hair Dryer', 'Makeup Kit'],
    'Sports & Outdoors': ['Yoga Mat', 'Dumbbell Set', 'Bicycle', 'Water Bottle']
}

data = []
for i in range(n_rows):
    order_id = f"ORD-2025-{1000 + i}"
    date = np.random.choice(dates)
    region = np.random.choice(regions, p=[0.30, 0.25, 0.25, 0.20])
    category = np.random.choice(list(categories.keys()))
    product = np.random.choice(categories[category])
    
    # Pricing logic per category
    base_price = {
        'Electronics': np.random.uniform(150, 1200),
        'Apparel': np.random.uniform(25, 180),
        'Home & Kitchen': np.random.uniform(40, 350),
        'Beauty & Care': np.random.uniform(20, 120),
        'Sports & Outdoors': np.random.uniform(15, 450)
    }[category]
    
    unit_price = round(base_price, 2)
    quantity = np.random.randint(1, 6)
    discount = np.random.choice([0.0, 0.05, 0.10, 0.15, 0.20], p=[0.5, 0.2, 0.15, 0.1, 0.05])
    gross_sales = round(unit_price * quantity, 2)
    net_sales = round(gross_sales * (1 - discount), 2)
    
    # Cost & Profit calculation (margin varies between 20% to 55%)
    margin_pct = np.random.uniform(0.22, 0.52)
    cost = round(net_sales * (1 - margin_pct), 2)
    profit = round(net_sales - cost, 2)
    
    payment_method = np.random.choice(['Credit Card', 'UPI / Wallet', 'Debit Card', 'Cash on Delivery'], p=[0.45, 0.30, 0.15, 0.10])
    
    data.append([order_id, date, region, category, product, unit_price, quantity, discount, gross_sales, net_sales, cost, profit, payment_method])

df = pd.DataFrame(data, columns=[
    'OrderID', 'OrderDate', 'Region', 'Category', 'Product',
    'UnitPrice', 'Quantity', 'DiscountRate', 'GrossSales', 'NetSales', 'Cost', 'Profit', 'PaymentMethod'
])

# Inject slight synthetic missingness for data cleaning demo
df.loc[df.sample(frac=0.015, random_state=42).index, 'PaymentMethod'] = np.nan

# Save raw dataset
df.to_csv("retail_sales_raw.csv", index=False)
print("✅ Raw dataset saved to 'retail_sales_raw.csv'. Rows:", len(df))

# -------------------------------------------------------------
# 2. DATA CLEANING & PREPROCESSING
# -------------------------------------------------------------
df_clean = df.copy()

# Handle missing values
df_clean['PaymentMethod'].fillna('Credit Card', inplace=True)

# Parse Dates & add helper features
df_clean['OrderDate'] = pd.to_datetime(df_clean['OrderDate'])
df_clean['Month'] = df_clean['OrderDate'].dt.strftime('%b')
df_clean['MonthNum'] = df_clean['OrderDate'].dt.month
df_clean['Quarter'] = 'Q' + df_clean['OrderDate'].dt.quarter.astype(str)

df_clean.to_csv("retail_sales_cleaned.csv", index=False)
print("✅ Cleaned dataset saved to 'retail_sales_cleaned.csv'.")

# -------------------------------------------------------------
# 3. KPI COMPUTATION
# -------------------------------------------------------------
total_revenue = df_clean['NetSales'].sum()
total_profit = df_clean['Profit'].sum()
total_orders = df_clean['OrderID'].nunique()
total_units = df_clean['Quantity'].sum()
avg_order_val = df_clean['NetSales'].mean()
overall_profit_margin = (total_profit / total_revenue) * 100

print("\n--- EXECUTIVE SUMMARY KEY METRICS ---")
print(f"Total Net Sales Revenue : ${total_revenue:,.2f}")
print(f"Total Gross Profit      : ${total_profit:,.2f}")
print(f"Overall Profit Margin   : {overall_profit_margin:.2f}%")
print(f"Total Orders Count      : {total_orders:,}")
print(f"Total Units Sold        : {total_units:,}")
print(f"Average Transaction Val : ${avg_order_val:,.2f}")

# -------------------------------------------------------------
# 4. DATA VISUALIZATION & CHART GENERATION
# -------------------------------------------------------------
os.makedirs("output_charts", exist_ok=True)

# Chart 1: Category Sales & Profit Comparison
cat_summary = df_clean.groupby('Category')[['NetSales', 'Profit']].sum().reset_index().sort_values(by='NetSales', ascending=False)
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(cat_summary))
width = 0.35

ax.bar(x - width/2, cat_summary['NetSales']/1e3, width, label='Net Revenue ($K)', color='#2563eb')
ax.bar(x + width/2, cat_summary['Profit']/1e3, width, label='Gross Profit ($K)', color='#10b981')

ax.set_ylabel('Amount in Thousands ($K)', fontsize=11, fontweight='bold')
ax.set_title('Revenue & Profitability by Product Category', fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(cat_summary['Category'], fontsize=10)
ax.legend(frameon=True, facecolor='white')
plt.tight_layout()
plt.savefig('output_charts/category_performance.png', dpi=300)
plt.close()

# Chart 2: Monthly Trend Analysis
monthly_trend = df_clean.groupby(['MonthNum', 'Month'])[['NetSales', 'Profit']].sum().reset_index().sort_values('MonthNum')
fig, ax = plt.subplots(figsize=(11, 5))

ax.plot(monthly_trend['Month'], monthly_trend['NetSales']/1e3, marker='o', linewidth=2.5, color='#1e3a8a', label='Revenue ($K)')
ax.fill_between(monthly_trend['Month'], monthly_trend['NetSales']/1e3, color='#3b82f6', alpha=0.15)
ax.plot(monthly_trend['Month'], monthly_trend['Profit']/1e3, marker='s', linewidth=2, color='#059669', label='Profit ($K)')

ax.set_title('Monthly Sales Revenue & Profit Trend (2025)', fontsize=14, fontweight='bold', pad=15)
ax.set_ylabel('Amount in Thousands ($K)', fontsize=11, fontweight='bold')
ax.legend(loc='upper left', frameon=True)
plt.tight_layout()
plt.savefig('output_charts/monthly_trend.png', dpi=300)
plt.close()

# Chart 3: Regional Sales Distribution
reg_sales = df_clean.groupby('Region')['NetSales'].sum()
fig, ax = plt.subplots(figsize=(6, 6))
colors = ['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6']

wedges, texts, autotexts = ax.pie(reg_sales, labels=reg_sales.index, autopct='%1.1f%%', startangle=140, colors=colors, wedgeprops=dict(width=0.4, edgecolor='white', linewidth=2))
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_weight('bold')

ax.set_title('Regional Sales Revenue Contribution', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('output_charts/regional_distribution.png', dpi=300)
plt.close()

print("✅ All visual charts generated and saved in 'output_charts/' directory.")