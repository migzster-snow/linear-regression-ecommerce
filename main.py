import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('ecommerce.csv')
 
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

# Exploratory Data Analysis (EDA)

# Jointplots
sns.jointplot(x="Time on Website", y="Yearly Amount Spent", data=df, alpha=0.5)
sns.jointplot(x="Time on App", y="Yearly Amount Spent", data=df, alpha=0.5)
sns.jointplot(x="Avg. Session Length", y="Yearly Amount Spent", data=df, alpha=0.5)
sns.jointplot(x="Length of Membership", y="Yearly Amount Spent", data=df, alpha=0.5)

plt.show()