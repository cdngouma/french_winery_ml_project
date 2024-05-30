# Wine Company Market Analysis and Forecasting
## Project Overview
This project is divided into two main parts:
1. Customer Segmentation: Analyzing customer data to identify distinct segments within the US market.
2. Forecasting Models: Developing predictive models to forecast key metrics such as wine consumption, sales, and gross margins.
The goal is to provide actionable insights to help the wine company effectively enter and compete in the US market.

## Customer Segmentation
The first part of the project focuses on customer segmentation to understand the different types of customers in the US market. This helps tailor marketing strategies and product offerings to different segments.

### Data Description
The dataset "marketing_campaign" from [Kaggle](https://www.kaggle.com/datasets/rodsaldanha/arketing-campaign), published in 2020, was used for this analysis. It contains data about a marketing campaign, including demographic and purchasing information for customers. The following columns were preselected for relevance to our study:
* Year_Birth, Education, Marital_Status, `Income`: Demographic information of the customers
* Kidhome, Teenhome: Household information
* Dt_Customer: The date when the customer was enrolled, which can be used to calculate customer tenure
* Recency: The number of days since the last purchase, indicating customer engagement and loyalty.
* MntWines: Amount spent on wine
* NumDealsPurchases: Number of purchases made with a discount
* NumCatalogPurchases: Number of purchases using a catalog
* NumStorePurchases: Number of purchases made in-store
* NumWebVisitsMonth: Number of visits to the company's website

#### Data Preprocessing
* **Handling Missing Values:** Missing values in the Income column were addressed by imputation.
* **Data Cleaning:** Irrelevant columns were dropped, and the data types were converted as necessary.
* **Feature Engineering:** New features were created, including Age, Household_Size, Pct_of_Purchases, Income_per_Household_Size, Online_Store_Ratio, TotalNumPurchases, and Customer_Tenure.
* **Correlation Analysis:** The relationship between each variable and MntWines was analyzed using the correlation coefficient.

### Exploratory Data Analysis
#### Correlation Analysis:
Numerical variables were analyzed for their correlation with MntWines, with key findings including high correlations for Income (0.686), Pct_of_Purchases (0.925), and TotalNumPurchases (0.756).
The full correlation table is detailed in the figure below.

#### ANOVA Tests:
* Marital_Status and MntWines: Not significant (p-value = 0.278)
* Education and MntWines: Significant (p-value = 1.21e-20)
* Income_Level and MntWines: Highly significant (p-value = 1.76e-201)

#### T-test:
* Kidhome and MntWines: Significant (p-value = 4.39e-149)
* Teenhome and MntWines: Not significant (p-value = 0.906)

### Clustering
Based on the analysis, the following variables were selected for clustering: Age, Income, Household_Size, TotalNumPurchases, Customer_Tenure, Education, and Kidhome. Two clustering algorithms were tested:
* DBSCAN (Silhouette Score = 0.335)
* KMeans (Silhouette Score = 0.669)

#### Results
KMeans produced 3 clusters with characteristics summarized in the figure below.

## Insights from Customer Segmentation
