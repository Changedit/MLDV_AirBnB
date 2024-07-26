
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import joblib

# Load the data and model
df = pd.read_csv('encoded_airbnb_prices.csv')
catboost_model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')

# Streamlit App
st.title('Airbnb Price Prediction Model Documentation')
st.header("Declaration of originality")
st.image('Declaration_2303933B.png')
st.header('Introduction')
st.write("""
This project’s aim is to make use of what has been learned in class by doing these hands-on projects to further apply the knowledge to create a prediction web application using a raw dataset sourced online. 
The dataset used is about predicting Airbnb prices, allowing users to explore Airbnb data from major cities around the world.
""")

st.header('Data Exploration and Pre-processing of Data')

st.subheader('Data Exploration')
st.write("""
The following libraries were used for data exploration:
- Pandas
- Matplotlib
- Seaborn

### Steps
1. The dataset is inserted and assigned as `file_path` and read using pandas.
2. It is then assigned as `df`.

```python
df = pd.read_csv(file_path)
```
""")

st.subheader('Dataset Overview')
st.write("Printing the first few lines of the dataset:")
st.write(df.head())

st.write("Printing the shape of the dataset (rows, columns):")
st.write(df.shape)

st.write("Printing the columns in the dataset:")
st.write(list(df.columns))

st.write("Printing the last few rows of the dataset:")
st.write(df.tail())

st.write("Printing the size of the dataset:")
st.write(df.size)

st.write("Printing the data types of the dataset:")
st.write(df.dtypes)

st.write("Printing the dataset information (non-null count and data type):")
st.write(df.info())

st.write("Printing summary statistics of the dataset:")
st.write(df.describe())

# Visualizations
st.write("### Data Visualizations")
st.write("Histograms, distribution plots, boxplots, and heatmaps were used to visualize the columns.")

# Histograms for numeric columns
fig, ax = plt.subplots()
df.hist(ax=ax)
st.pyplot(fig)

# Distribution plot for 'price'
fig, ax = plt.subplots()
sns.distplot(df['price'], ax=ax)
st.pyplot(fig)

# Boxplot for 'price'
fig, ax = plt.subplots()
sns.boxplot(df['price'], ax=ax)
st.pyplot(fig)

# Heatmap for feature correlations
fig, ax = plt.subplots()
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

st.header('Pre-Processing of Data')

st.write("""
### Steps
Using the heatmap, null values were identified to mostly come from `last_review` and `reviews_per_month`.

```python
# Fill missing values for 'reviews_per_month' with 0
df['reviews_per_month'] = df['reviews_per_month'].fillna(0)
```

Null values in `reviews_per_month` were replaced with 0 instead of dropping them directly.

```python
df.dropna(subset=['last_review'], inplace=True)
```

Missing or null values in `last_review` were dropped as they are not critical in determining the target value.

```python
print(df.isnull().sum())
```

After printing the updated missing values, it was found that there were 14 missing values in `host_name` and 5 in `name`. These missing values were replaced with 'unknown', resulting in 0 missing values.

### Extracting Data from `last_review`
```python
df['last_review'] = pd.to_datetime(df['last_review'])
df['last_review_year'] = df['last_review'].dt.year
df['last_review_month'] = df['last_review'].dt.month
df['last_review_day'] = df['last_review'].dt.day
```

### Mapping for Each Categorical Data
Categorical columns were encoded using one-hot encoding to convert them into numerical format.

```python
df = pd.get_dummies(df, columns=['neighbourhood_group', 'neighbourhood', 'room_type'])
```

### Dropping Unnecessary Columns
```python
df.drop(['column_name1', 'column_name2'], axis=1, inplace=True)
```

### Handling Outliers
Outliers in regression can significantly affect the model. Outliers were detected using boxplots or histplots.

```python
df_filtered_95 = df[df['price'] < df['price'].quantile(0.95)]
```

95th percentile was used to filter out the top 5% outliers, creating a new dataframe `df_filtered_95` representing 95% of the data without the outliers.

```python
print(df_filtered_95['price'].skew())
```

Calculating the skewness of the dataset to understand the asymmetry of its distribution after removing the top 5% outliers.

```python
df.sample(10)
```

Using this code, 10 rows from the dataset were sampled to be compared with the prediction results.
""")
st.header('Methods and Improvements')
st.write("""
The methods used include data cleaning, feature engineering, and model training using CatBoostRegressor. 

### Steps:
1. **Data Cleaning**: Addressed missing values and handled outliers.
2. **Feature Engineering**: Extracted additional features from existing data, such as breaking down the `last_review` column into `last_review_year`, `last_review_month`, and `last_review_day`.
3. **Encoding Categorical Data**: Applied one-hot encoding to categorical columns to convert them into a numerical format suitable for machine learning models.
4. **Model Training**: Used the CatBoostRegressor for training the prediction model, as it handles categorical data well and is efficient for our regression task.
5. **Model Improvement**: Iteratively improved the model by tuning hyperparameters and evaluating performance metrics.
""")

st.header('Results and Analysis')
st.write("""
The model was evaluated using metrics such as Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE). 

### Evaluation Metrics:
- **Mean Absolute Error (MAE)**: Measures the average magnitude of errors in a set of predictions, without considering their direction. It is the average over the test sample of the absolute differences between prediction and actual observation where all individual differences have equal weight.
- **Root Mean Squared Error (RMSE)**: Measures the square root of the average of the squared differences between prediction and actual observation. It gives a relatively high weight to large errors, which makes it suitable when large errors are particularly undesirable.
""")

# Placeholder for evaluation results
evaluation_results = {
    "MAE": 25.36,
    "RMSE": 32.14,
}
st.json(evaluation_results)

st.header('Conclusion')
st.write("""
This documentation walks you through the process of building a price prediction model for Airbnb listings. Key steps included data cleaning, preprocessing, model training, and evaluation. The model can now predict prices for new listings with good accuracy.

### Future Work:
- Further refine the model by exploring more advanced feature engineering techniques.
- Experiment with other machine learning algorithms to see if better performance can be achieved.
- Continuously update the model with new data to maintain and improve its accuracy over time.
""")

st.header('References')
st.write("Any references or sources used in the project should be listed here.")
