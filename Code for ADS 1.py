import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import skew, kurtosis

# Load the dataset
file_path = 'project1_df.csv'  # Update this path as needed
data = pd.read_csv(file_path)

# Data Cleaning
data['Purchase Date'] = pd.to_datetime(data['Purchase Date'], errors='coerce')
data['Year'] = data['Purchase Date'].dt.year  # For year-based analysis

# Statistical Summary Function
def get_statistical_summary(data):
    # Basic descriptive statistics
    summary_stats = data.describe()
    # Correlation matrix
    correlation_matrix = data.corr(numeric_only=True)
    # Skewness and kurtosis for each numerical column
    skewness = data.select_dtypes(include=[np.number]).apply(lambda x: skew(x.dropna()))
    kurt = data.select_dtypes(include=[np.number]).apply(lambda x: kurtosis(x.dropna()))
    
    print("Summary Statistics:\n", summary_stats)
    print("\nCorrelation Matrix:\n", correlation_matrix)
    print("\nSkewness:\n", skewness)
    print("\nKurtosis:\n", kurt)

# Plot 1: Pie Chart for Purchase Method
def plot_purchase_method_pie(data):
    purchase_counts = data['Purchase Method'].value_counts()
    plt.figure(figsize=(8, 8))
    wedges, texts, autotexts = plt.pie(
        purchase_counts, labels=purchase_counts.index, autopct='%1.1f%%', startangle=140, 
        textprops={'weight':'bold'}, pctdistance=0.85
    )
    plt.setp(autotexts, size=10, weight="bold")
    plt.title("Distribution of Purchase Methods", fontsize=14, fontweight='bold')
    plt.annotate("Breakdown of payment preferences", 
                 xy=(0, 0), xytext=(1.2, -1.2), 
                 textcoords='axes fraction', fontsize=10, fontweight='bold', ha="right")
    plt.show()

# Plot 2: Line Chart for Annual Sales Trend with Trendline
def plot_annual_sales_trend_with_trendline(data):
    annual_sales = data.groupby('Year')['Net Amount'].sum()
    years = annual_sales.index
    sales_values = annual_sales.values
    
    plt.figure(figsize=(12, 6))
    plt.plot(years, sales_values, marker='o', color='b', linestyle='-', linewidth=1.5, markersize=4, label="Annual Sales")

    # Calculate and add a trendline
    z = np.polyfit(years, sales_values, 1)
    p = np.poly1d(z)
    plt.plot(years, p(years), linestyle='--', color="red", linewidth=1.5, label="Trendline")

    plt.xticks(years, rotation=45, fontsize=10)
    plt.title("Annual Sales Trend (Net Amount) with Trendline", fontsize=14, fontweight='bold')
    plt.xlabel("Year", fontsize=12, fontweight='bold')
    plt.ylabel("Total Net Amount (INR)", fontsize=12, fontweight='bold')
    plt.grid(visible=True, linestyle='--', alpha=0.5)

    # Highlight the highest sales year
    max_sales_year = years[sales_values.argmax()]
    max_sales_value = sales_values.max()
    plt.annotate(f"Peak: {max_sales_value:.0f} INR", 
                 xy=(max_sales_year, max_sales_value), 
                 xytext=(0.8, 0.9), textcoords='axes fraction', 
                 arrowprops=dict(facecolor='grey', shrink=0.05), 
                 fontsize=10, fontweight='bold', ha='center')

    plt.legend()
    plt.show()

# Plot 3: Correlation Heatmap
def plot_correlation_heatmap(data):
    plt.figure(figsize=(8, 6))
    correlation_matrix = data.corr(numeric_only=True)
    heatmap = sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", 
                          annot_kws={"weight": "bold"}, cbar_kws={'shrink': 0.8})
    plt.title("Correlation Matrix Heatmap", fontsize=14, fontweight='bold')
    plt.xticks(fontsize=10, fontweight='bold', rotation=45)
    plt.yticks(fontsize=10, fontweight='bold', rotation=0)

    # Focused annotation on the strongest correlation
    max_corr_value = correlation_matrix.loc["Gross Amount", "Net Amount"]
    plt.text(2, 3, f"Strong\ncorrelation\n({max_corr_value:.2f})", 
             fontsize=9, fontweight='bold', color='black', ha='center', va='center')
    plt.show()

# Box Plot visualization

def plot_age_group_net_amount_box(data):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Age Group', y='Net Amount', data=data, palette="Set2")
    plt.title("Net Amount Distribution by Age Group", fontsize=14, fontweight='bold')
    plt.xlabel("Age Group", fontsize=12, fontweight='bold')
    plt.ylabel("Net Amount (INR)", fontsize=12, fontweight='bold')
    
    # Adding annotation to highlight the median for each age group
    medians = data.groupby('Age Group')['Net Amount'].median()
    for idx, median in enumerate(medians):
        plt.text(idx, median, f'{median:.0f}', ha='center', va='center', fontweight='bold', color='black')
    plt.show()

# Running the box plot function
plot_age_group_net_amount_box(data)

# Run all functions for analysis and visualization
get_statistical_summary(data)
plot_purchase_method_pie(data)
plot_annual_sales_trend_with_trendline(data)
plot_correlation_heatmap(data)
plot_age_group_net_amount_box(data)