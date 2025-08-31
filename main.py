# Import necessary libraries
# pandas for data manipulation and analysis
import pandas as pd
# matplotlib for creating static, animated, and interactive visualisations
import matplotlib.pyplot as plt
# seaborn for statistical data visualisation based on matplotlib
import seaborn as sns

# Load the dataset from a CSV file into a pandas DataFrame
# A DataFrame is a 2-dimensional labeled data structure with columns of potentially different types
df = pd.read_csv('ecommerce.csv')

# Display the first few rows of the dataset to understand its structure
# This helps us get a quick overview of the data
# head = df.head()
# print(head.to_string())

"""
                           Email                                                     Address            Avatar  Avg. Session Length  Time on App  Time on Website  Length of Membership  Yearly Amount Spent
0      mstephenson@fernandez.com                835 Frank Tunnel\nWrightmouth, MI 82180-9605            Violet            34.497268    12.655651        39.577668              4.082621           587.951054
1              hduke@hotmail.com              4547 Archer Common\nDiazchester, CA 06566-8576         DarkGreen            31.926272    11.109461        37.268959              2.664034           392.204933
2               pallen@yahoo.com  24645 Valerie Unions Suite 582\nCobbborough, DC 99414-7564            Bisque            33.000915    11.330278        37.110597              4.104543           487.547505
3        riverarebecca@gmail.com            1414 David Throughway\nPort Jason, OH 22070-1220       SaddleBrown            34.305557    13.717514        36.721283              3.120179           581.852344
4  mstephens@davidson-herman.com     14023 Rodriguez Passage\nPort Jacobville, PR 37242-1057  MediumAquaMarine            33.330673    12.795189        37.536653              4.446308           599.406092
"""

# Get information about the dataset including data types and non-null values
# This helps identify missing values and understand the data structure
# info = df.info()

"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 500 entries, 0 to 499
Data columns (total 8 columns):
 #   Column                Non-Null Count  Dtype
---  ------                --------------  -----
 0   Email                 500 non-null    object
 1   Address               500 non-null    object
 2   Avatar                500 non-null    object
 3   Avg. Session Length   500 non-null    float64
 4   Time on App           500 non-null    float64
 5   Time on Website       500 non-null    float64
 6   Length of Membership  500 non-null    float64
 7   Yearly Amount Spent   500 non-null    float64
dtypes: float64(5), object(3)
memory usage: 31.4+ KB
"""

# Generate descriptive statistics of the numerical columns
# This includes count, mean, standard deviation, min, max, and quartiles
# describe = df.describe()
# print(describe.to_string())

"""
       Avg. Session Length  Time on App  Time on Website  Length of Membership  Yearly Amount Spent
count           500.000000   500.000000       500.000000            500.000000           500.000000
mean             33.053194    12.052488        37.060445              3.533462           499.314038
std               0.992563     0.994216         1.010489              0.999278            79.314782
min              29.532429     8.508152        33.913847              0.269901           256.670582
25%              32.341822    11.388153        36.349257              2.930450           445.038277
50%              33.082008    11.983231        37.069367              3.533975           498.887875
75%              33.711985    12.753850        37.716432              4.126502           549.313828
max              36.139662    15.126994        40.005182              6.922689           765.518462
"""

# Exploratory Data Analysis (EDA) - Understanding relationships in data through visualisation

# Jointplots show the relationship between two variables along with their distributions
# Uncomment any of these to see the specific relationships

# Relationship between Time on Website and Yearly Amount Spent
# sns.jointplot(x="Time on Website", y="Yearly Amount Spent", data=df, alpha=0.5)
# plt.show()

# Relationship between Time on App and Yearly Amount Spent
# sns.jointplot(x="Time on App", y="Yearly Amount Spent", data=df, alpha=0.5)
# plt.show()

# Relationship between Avg. Session Length and Yearly Amount Spent
# sns.jointplot(x="Avg. Session Length", y="Yearly Amount Spent", data=df, alpha=0.5)
# plt.show()

# Relationship between Length of Membership and Yearly Amount Spent
# sns.jointplot(x="Length of Membership", y="Yearly Amount Spent", data=df, alpha=0.5)
# plt.show()

# Pairplot creates a grid of scatterplots showing relationships between all numerical variables
# sns.pairplot(df, kind="scatter", plot_kws={'alpha':0.5})
# plt.show()

# Linear Model Plots (lmplots) show the linear relationship between variables with a regression line
# sns.lmplot(
#     x="Length of Membership", 
#     y="Yearly Amount Spent", 
#     data=df,
#     scatter_kws={'alpha':0.5}
# )
# plt.show()

# sns.lmplot(
#     x="Avg. Session Length",
#     y="Yearly Amount Spent",
#     data=df,
#     scatter_kws={'alpha':0.5}
# )
# plt.show()

# sns.lmplot(
#     x="Time on App",
#     y="Yearly Amount Spent",
#     data=df,
#     scatter_kws={'alpha':0.5}
# )
# plt.show()

# sns.lmplot(
#     x="Time on Website",
#     y="Yearly Amount Spent",
#     data=df,
#     scatter_kws={'alpha':0.5}
# )
# plt.show()

# Train-Test Split: Dividing data into training and testing sets
# This helps evaluate how well our model generalises to new data
from sklearn.model_selection import train_test_split

# Select features (independent variables) and target (dependent variable)
# We're using numerical features that might influence yearly spending
X = df[["Avg. Session Length", "Time on App", "Time on Website", "Length of Membership"]]
y = df["Yearly Amount Spent"]

# Display the features and target (uncomment to see)
# print(X.to_string())
# print(y.to_string())

# Split the data: 70% for training, 30% for testing
# random_state ensures reproducible results
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Display the split data (uncomment to see)
# print(X_train.to_string())
# print(X_test.to_string())
# print(y_train.to_string())
# print(y_test.to_string())

# Create and train a Linear Regression model
from sklearn.linear_model import LinearRegression

# Initialise the model
lm = LinearRegression()

# Train the model using the training data
lm.fit(X_train, y_train)

# Display the model coefficients (uncomment to see)
# print(lm.coef_)

# Create a DataFrame to show the coefficient for each feature
# Coefficients indicate how much the target variable changes with a one-unit change in each feature
cdf = pd.DataFrame(lm.coef_, X.columns, columns=['Coefficient'])
# print(cdf.to_string())

"""
                      Coefficient
Avg. Session Length     25.724256
Time on App             38.597135
Time on Website          0.459148
Length of Membership    61.674732
"""

# Make predictions using the trained model on the test data
predictions = lm.predict(X_test)

# Create a scatter plot to compare predictions with actual values
# A good model would have points close to a diagonal line
# sns.scatterplot(x=predictions, y=y_test)
# plt.xlabel("Predictions")
# plt.title("Evaluation of Model")
# plt.show()

# Evaluate the model performance using various metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

# Calculate and display error metrics (uncomment to see)
# print('Mean Absolute Error: ', mean_absolute_error(y_test, predictions))
# print('Mean Squared Error: ', mean_squared_error(y_test, predictions))
# print('Root Mean Squared Error: ', math.sqrt(mean_squared_error(y_test, predictions)))

"""
Mean Absolute Error:  8.426091641432116
Mean Squared Error:  103.91554136503333
Root Mean Squared Error:  10.193897260863155
"""

# Analyse residuals (differences between actual and predicted values)
# Residuals should be randomly distributed for a good model
residuals = y_test - predictions

# Create a distribution plot of residuals
# Ideally, residuals should be normally distributed around zero
sns.displot(residuals, bins=30, kde=True)
plt.title("Residuals Distribution")
plt.show()

# Create a probability plot to check if residuals are normally distributed
# Points should follow the red line for normal distribution
import pylab
import scipy.stats as stats

stats.probplot(residuals, dist="norm", plot=pylab)
pylab.title("Q-Q Plot of Residuals")
pylab.show()