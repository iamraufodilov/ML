# load librries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# load dataset
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
print(data['feature_names']) # as you can see from the output our data has nearly 30 feautre, it means data has 30 dimensions

# convert sklearn class data to pd dataframe
df = pd.DataFrame(data['data'], columns=data['feature_names'])

# scale data
scaler = StandardScaler()
scaler.fit(df)
scaled_data = scaler.transform(df)

# import PCA
pca_model = PCA(n_components=3) # here pay attention we allocate n_components to 3 it means we are only interested in 3 PCA 
# with assuming first 3 PCA holds much data varience
pca_model.fit(scaled_data)
result_pca = pca_model.transform(scaled_data)
print(result_pca.shape) # as you can see from the result now our data has three dimesions which shrinked from 30 dimesnsion

# to see pca components
print("Here is value of PCA: \n",pca_model.components_)

# to get variance values for components 
print("Here is variance ratio for all components: \n",pca_model.explained_variance_ratio_)
# here we go we manage to illustrate the pca on the dataset
# in case we diminish dataset dimension from 30 to 3 NOTE we assumed first 3 components holds most data variance
# actually there is no clue to chose 3 you can choose 2 or 4 5 
# but from the variance ratio you can see even PC3 holds less amount of ration

# thank you 
# rauf odilov