# Loading all the required packages and libraries

import os
os.chdir('C:\Users\shardul\Desktop\Rashmi\Clustering\KNN')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


from sklearn import metrics
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

cust_data =pd.read_excel("cust_data.xlsx")

cust_data.shape # (30000, 38)
cust_data.info()
'''
Data columns (total 38 columns):
 #   Column                  Non-Null Count  Dtype 
---  ------                  --------------  ----- 
 0   Cust_ID                 30000 non-null  int64 
 1   Gender                  27276 non-null  object
 2   Orders                  30000 non-null  int64 
 3   Jordan                  30000 non-null  int64 
 4   Gatorade                30000 non-null  int64 
 5   Samsung                 30000 non-null  int64 
 6   Asus                    30000 non-null  int64 
 7   Udis                    30000 non-null  int64 
 8   Mondelez International  30000 non-null  int64 
 9   Wrangler                30000 non-null  int64 
 10  Vans                    30000 non-null  int64 
 11  Fila                    30000 non-null  int64 
 12  Brooks                  30000 non-null  int64 
 13  H&M                     30000 non-null  int64 
 14  Dairy Queen             30000 non-null  int64 
 15  Fendi                   30000 non-null  int64 
 16  Hewlett Packard         30000 non-null  int64 
 17  Pladis                  30000 non-null  int64 
 18  Asics                   30000 non-null  int64 
 19  Siemens                 30000 non-null  int64 
 20  J.M. Smucker            30000 non-null  int64 
 21  Pop Chips               30000 non-null  int64 
 22  Juniper                 30000 non-null  int64 
 23  Huawei                  30000 non-null  int64 
 24  Compaq                  30000 non-null  int64 
 25  IBM                     30000 non-null  int64 
 26  Burberry                30000 non-null  int64 
 27  Mi                      30000 non-null  int64 
 28  LG                      30000 non-null  int64 
 29  Dior                    30000 non-null  int64 
 30  Scabal                  30000 non-null  int64 
 31  Tommy Hilfiger          30000 non-null  int64 
 32  Hollister               30000 non-null  int64 
 33  Forever 21              30000 non-null  int64 
 34  Colavita                30000 non-null  int64 
 35  Microsoft               30000 non-null  int64 
 36  Jiffy mix               30000 non-null  int64 
 37  Kraft                   30000 non-null  int64 
dtypes: int64(37), object(1)
'''
cust_data.isnull().sum() # only Gender has 2724 missing values


## Variable 1 - Gender

cust_data.Gender.value_counts()
'''
cust_data.Gender.value_counts()
Out[14]: 
F    22054
M     5222
Name: Gender, dtype: int64
'''
(5222/22054) * 100
sns.countplot(cust_data.Gender)

# since this is a categorical variable euclidean distance would be non-sensical
# hence we wont consider Gender in our cluster analysis

# Variable 2 - Orders - continuous
cust_data.Orders.describe() # count 30000
'''
Out[20]: 
count    30000.000000
mean         4.169800
std          3.590311
min          0.000000
25%          1.000000
50%          4.000000
75%          7.000000
max         12.000000
Name: Orders, dtype: float64
'''
#histogram
plt.hist(cust_data.Orders)

# boxplot
plt.boxplot(cust_data.Orders,vert=False)

# no missing and no outliers
# will be considered in cluster analysis


# Variable 3 - Jordan

cust_data.Jordan.describe() # count 30000
cust_data.Jordan.isnull().sum() # no missing values
sns.countplot(cust_data.Jordan)
# to be included in cluster analysis

# Variable 4 - Gatorade

cust_data.Gatorade.describe() # count 30000
cust_data.Gatorade.isnull().sum() # no missing values
sns.countplot(cust_data.Gatorade)
# to be included in cluster analysis

# Variable 5 - Samsung

cust_data.Samsung.describe() # count 30000
cust_data.Samsung.isnull().sum() # no missing values
sns.countplot(cust_data.Samsung)
# to be included in cluster analysis

# Variable 6 - Asus

cust_data.Asus.describe() # count 30000
cust_data.Asus.isnull().sum() # no missing values
sns.countplot(cust_data.Asus)
# to be included in cluster analysis


# Variable 7 - Udis

cust_data.Udis.describe() # count 30000
cust_data.Udis.isnull().sum() # no missing values
plt.boxplot(cust_data.Udis)
sns.countplot(cust_data.Udis)
# to be included in cluster analysis


# Variable 8 - Mondelez International

# renaming the column properly
cust_data = cust_data.rename(columns = {'Mondelez International':'Mondelez_International'})
cust_data.Mondelez_International.describe() # count 30000
cust_data.Mondelez_International.isnull().sum() # no missing values

plt.boxplot(cust_data.Mondelez_International)
sns.countplot(cust_data.Mondelez_International)
# to be included in cluster analysis

# Variable 9 - Wrangler                

cust_data.Wrangler.describe() # count 30000
cust_data.Wrangler.isnull().sum() # no missing values
plt.boxplot(cust_data.Wrangler)
sns.countplot(cust_data.Wrangler)
# to be included in cluster analysis


# Variable 10 - Vans                                    

cust_data.Vans.describe() # count 30000
cust_data.Vans.isnull().sum() # no missing values
plt.boxplot(cust_data.Vans)
sns.countplot(cust_data.Vans)
# to be included in cluster analysis

# Variable 11 - Fila                                    

cust_data.Fila.describe() # count 30000
cust_data.Fila.isnull().sum() # no missing values
plt.boxplot(cust_data.Fila)
sns.countplot(cust_data.Fila)
# to be included in cluster analysis

# Variable 12 - Brooks                                                      

cust_data.Brooks.describe() # count 30000
cust_data.Brooks.isnull().sum() # no missing values
plt.boxplot(cust_data.Brooks)
sns.countplot(cust_data.Brooks)
# to be included in cluster analysis

# Variable 13 -  H&M                 
                                     
#renaming the variable properly
cust_data = cust_data.rename(columns = {'H&M':'HandM'})
cust_data.HandM.describe() # count 30000
cust_data.HandM.isnull().sum() # no missing values

plt.boxplot(cust_data.HandM)
sns.countplot(cust_data.HandM)
# to be included in cluster analysis

# Variable 14 -  Dairy Queen                
                                     

cust_data['Dairy Queen'].describe() # count 30000
cust_data.['Dairy Queen'].isnull().sum() # no missing values
plt.boxplot(cust_data['Dairy Queen'])
sns.countplot(cust_data['Dairy Queen'])
# to be included in cluster analysis

# Variable 15 -  Fendi                 
                                     
cust_data.Fendi.describe() # count 30000
cust_data.Fendi.isnull().sum() # no missing values
plt.boxplot(cust_data.Fendi)
sns.countplot(cust_data.Fendi)
# to be included in cluster analysis

# Variable 16 -  Hewlett Packard                
                                     
cust_data['Hewlett Packard'].describe() # count 30000
cust_data.['Hewlett Packard'].isnull().sum() # no missing values
plt.boxplot(cust_data['Hewlett Packard'])
sns.countplot(cust_data['Hewlett Packard'])
# to be included in cluster analysis

# Variable 16 -  Pladis                
                                     
cust_data['Pladis'].describe() # count 30000
cust_data.['Pladis'].isnull().sum() # no missing values
plt.boxplot(cust_data['Pladis'])
sns.countplot(cust_data['Pladis'])
# to be included in cluster analysis


# Variable 17 -  Asics                                   
                                     
cust_data['Asics'].describe() # count 30000
cust_data['Asics'].isnull().sum() # no missing values
plt.boxplot(cust_data['Asics'])
sns.countplot(cust_data['Asics'])
# to be included in cluster analysis


# Variable 18 -  Siemens                                                    
                                     
cust_data['Siemens'].describe() # count 30000
cust_data['Siemens'].isnull().sum() # no missing values
plt.boxplot(cust_data['Siemens'])
sns.countplot(cust_data['Siemens'])
# to be included in cluster analysis

# Variable 19 -  J.M. Smucker                                                    
                                     
cust_data['J.M. Smucker'].describe() # count 30000
cust_data['J.M. Smucker'].isnull().sum() # no missing values
plt.boxplot(cust_data['J.M. Smucker'])
sns.countplot(cust_data['J.M. Smucker'])
# to be included in cluster analysis


# Variable 20 -  Pop Chips                                                    
                                     
cust_data['Pop Chips'].describe() # count 30000
cust_data['Pop Chips'].isnull().sum() # no missing values
plt.boxplot(cust_data['Pop Chips'])
sns.countplot(cust_data['Pop Chips'])
# to be included in cluster analysis

# Variable 21 -  Juniper                                                                     
                                     
cust_data['Juniper'].describe() # count 30000
cust_data['Juniper'].isnull().sum() # no missing values
plt.boxplot(cust_data['Juniper'])
sns.countplot(cust_data['Juniper'])
# to be included in cluster analysis


# Variable 22 -  Huawei                                                                                       
                                     
cust_data['Huawei'].describe() # count 30000
cust_data['Huawei'].isnull().sum() # no missing values
plt.boxplot(cust_data['Huawei'])
sns.countplot(cust_data['Huawei'])
# to be included in cluster analysis


# Variable 23 -  Compaq                                                                                                         
                                     
cust_data['Compaq'].describe() # count 30000
cust_data['Compaq'].isnull().sum() # no missing values
plt.boxplot(cust_data['Compaq'])
sns.countplot(cust_data['Compaq'])
# to be included in cluster analysis

# Variable 23 -  IBM                                                                                                         
                                     
cust_data['IBM'].describe() # count 30000
cust_data['IBM'].isnull().sum() # no missing values
plt.boxplot(cust_data['IBM'])
sns.countplot(cust_data['IBM'])
# to be included in cluster analysis

# Variable 24 -  Burberry                                                                                                         
                                     
cust_data['Burberry'].describe() # count 30000
cust_data['Burberry'].isnull().sum() # no missing values
plt.boxplot(cust_data['Burberry'])
sns.countplot(cust_data['Burberry'])
# to be included in cluster analysis

# Variable 25 -  Mi                                                                                                         
                                     
cust_data['Mi'].describe() # count 30000
cust_data['Mi'].isnull().sum() # no missing values
plt.boxplot(cust_data['Mi'])
sns.countplot(cust_data['Mi'])
# to be included in cluster analysis


# Variable 26 -  LG                                                                                                         
                                     
cust_data['LG'].describe() # count 30000
cust_data['LG'].isnull().sum() # no missing values
plt.boxplot(cust_data['LG'])
sns.countplot(cust_data['LG'])
# to be included in cluster analysis

# Variable 27 -  Dior                                                                                                                             
                                     
cust_data['Dior'].describe() # count 30000
cust_data['Dior'].isnull().sum() # no missing values
plt.boxplot(cust_data['Dior'])
sns.countplot(cust_data['Dior'])
# to be included in cluster analysis

# Variable 28 -  Scabal                                                                                                                             
                                     
cust_data['Scabal'].describe() # count 30000
cust_data['Scabal'].isnull().sum() # no missing values
plt.boxplot(cust_data['Scabal'])
sns.countplot(cust_data['Scabal'])
# to be included in cluster analysis


# Variable 29 -  Tommy Hilfiger                                                                                                                             
                                     
cust_data['Tommy Hilfiger'].describe() # count 30000
cust_data['Tommy Hilfiger'].isnull().sum() # no missing values
plt.boxplot(cust_data['Tommy Hilfiger'])
sns.countplot(cust_data['Tommy Hilfiger'])
# to be included in cluster analysis

# Variable 30 -  Hollister                                                                                                                            
                                     
cust_data['Hollister'].describe() # count 30000
cust_data['Hollister'].isnull().sum() # no missing values
plt.boxplot(cust_data['Hollister'])
sns.countplot(cust_data['Hollister'])
# to be included in cluster analysis

# Variable 31 -  Forever 21                                                                                                                            
                                     
cust_data['Forever 21'].describe() # count 30000
cust_data['Forever 21'].isnull().sum() # no missing values
plt.boxplot(cust_data['Forever 21'])
sns.countplot(cust_data['Forever 21'])
# to be included in cluster analysis

# Variable 32 -  Colavita                                                                                                                                            
                                     
cust_data['Colavita'].describe() # count 30000
cust_data['Colavita'].isnull().sum() # no missing values
plt.boxplot(cust_data['Colavita'])
sns.countplot(cust_data['Colavita'])
# to be included in cluster analysis


# Variable 33 -  Microsoft                                                                                                                                             
                                     
cust_data['Microsoft'].describe() # count 30000
cust_data['Microsoft'].isnull().sum() # no missing values
plt.boxplot(cust_data['Microsoft'])
sns.countplot(cust_data['Microsoft'])
# to be included in cluster analysis

# Variable 34 -  Jiffy mix                                                                                                                                             
                                     
cust_data['Jiffy mix'].describe() # count 30000
cust_data['Jiffy mix'].isnull().sum() # no missing values
plt.boxplot(cust_data['Jiffy mix'])
sns.countplot(cust_data['Jiffy mix'])
# to be included in cluster analysis

# Variable 35 -  Kraft                                                                                                                                                                
                                     
cust_data['Kraft'].describe() # count 30000
cust_data['Kraft'].isnull().sum() # no missing values
plt.boxplot(cust_data['Kraft'])
sns.countplot(cust_data['Kraft'])
# to be included in cluster analysis

df1 = cust_data
df1.info()

# remove CUST_ID, Gender and make df2
df2 = df1.iloc[:,2:]
df2.info()

df2.to_csv('df2.csv')

# standardisation and make df3
from sklearn.preprocessing import StandardScaler 
sc_x = StandardScaler() 
df3 = sc_x.fit_transform(df2)  

# the standardisation will give array hence convert it to dataframe again

df3 = pd.DataFrame(df3)
df3.columns = df2.columns
df3.info()


df3.to_csv('df3.csv')

df3 = pd.read_csv('df3.csv')

# find optimal clusters

wcss = []
for i in range(1,7):
    kmeans = KMeans(n_clusters = i, init = 'random', random_state = 42)
    kmeans.fit(df3)
    wcss.append(kmeans.inertia_)
    
print(wcss)
#[2250001077500.0522, 562503063703.6266, 250005935728.44418, 140634006709.0325, 90012127407.28387, 62507026740.16316]

plt.plot(range(1,7), wcss, 'bx-')
plt.title('The Elbow Method')
plt.xlabel('No of clusters')
plt.ylabel('wcss')
plt.show()


'''
based on this we can select 2 clusters as optimal since distance 
between 1 and 2 clusters is maximum and after 2 the distance between 2 and 3 ,
3 and 4 , 4 and 5  and 5 and 6 seems the same
'''



# A list holds the silhouette coefficients for each k
silhouette_coefficients = []
for k in range(2,6):
     kmeans = KMeans(n_clusters=k)
     kmeans.fit(df3)
     score = silhouette_score(df3, kmeans.labels_)
     silhouette_coefficients.append(score)
     
print(silhouette_coefficients)
'''
[0.6267069733735835, 0.5901527041176605, 0.5718633311082669, 0.5608966031983033]
'''     
plt.plot(range(2,6), silhouette_coefficients)
plt.xticks(range(2,6))
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Coefficient")
plt.show()


# silhoutte score is highest for number of clusters as 2
# 2 is the optimal number of clusters

kmeans = KMeans(n_clusters=2)
clusters = kmeans.fit_predict(df3)
clusters
# Out[20]: array([0, 0, 0, ..., 1, 1, 1])

#Cluster names as 0 and 1 is not comfortable so converting them to 1 and 2
final_cluster = clusters+1
final_cluster
#Out[23]: array([1, 1, 1, ..., 2, 2, 2])

#Coverting arrays to list
final_cluster = list(final_cluster)
type(final_cluster) # list

df3['Clusters'] = final_cluster
df3.info()
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 30000 entries, 0 to 29999
Data columns (total 38 columns):
 #   Column                  Non-Null Count  Dtype  
---  ------                  --------------  -----  
 0   Unnamed: 0              30000 non-null  int64  
 1   Orders                  30000 non-null  float64
 2   Jordan                  30000 non-null  float64
 3   Gatorade                30000 non-null  float64
 4   Samsung                 30000 non-null  float64
 5   Asus                    30000 non-null  float64
 6   Udis                    30000 non-null  float64
 7   Mondelez_International  30000 non-null  float64
 8   Wrangler                30000 non-null  float64
 9   Vans                    30000 non-null  float64
 10  Fila                    30000 non-null  float64
 11  Brooks                  30000 non-null  float64
 12  H&M                     30000 non-null  float64
 13  Dairy Queen             30000 non-null  float64
 14  Fendi                   30000 non-null  float64
 15  Hewlett Packard         30000 non-null  float64
 16  Pladis                  30000 non-null  float64
 17  Asics                   30000 non-null  float64
 18  Siemens                 30000 non-null  float64
 19  J.M. Smucker            30000 non-null  float64
 20  Pop Chips               30000 non-null  float64
 21  Juniper                 30000 non-null  float64
 22  Huawei                  30000 non-null  float64
 23  Compaq                  30000 non-null  float64
 24  IBM                     30000 non-null  float64
 25  Burberry                30000 non-null  float64
 26  Mi                      30000 non-null  float64
 27  LG                      30000 non-null  float64
 28  Dior                    30000 non-null  float64
 29  Scabal                  30000 non-null  float64
 30  Tommy Hilfiger          30000 non-null  float64
 31  Hollister               30000 non-null  float64
 32  Forever 21              30000 non-null  float64
 33  Colavita                30000 non-null  float64
 34  Microsoft               30000 non-null  float64
 35  Jiffy mix               30000 non-null  float64
 36  Kraft                   30000 non-null  float64
 37  Clusters                30000 non-null  int64  
dtypes: float64(36), int64(2)
'''

df3.Orders.groupby(df3.Clusters).describe()
'''
Out[30]: 
            count      mean       std  ...       50%       75%       max
Clusters                               ...                              
1         14990.0 -0.023083  1.008748  ... -0.047295  0.788301  2.180962
2         15010.0  0.023053  0.990717  ... -0.047295  0.788301  2.180962

[2 rows x 8 columns]
'''

clust_profile = df3.groupby('Clusters').mean()
clust_profile
'''
Out[32]: 
          Unnamed: 0    Orders    Jordan  ...  Microsoft  Jiffy mix     Kraft
Clusters                                  ...                                
1             7494.5 -0.023083 -0.065721  ...   0.005328   0.017941 -0.030404
2            22494.5  0.023053  0.065634  ...  -0.005321  -0.017917  0.030363

[2 rows x 37 columns]
'''