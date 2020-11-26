from Core.data_encoding_sampling import *
from sklearn.cluster import KMeans


data_encoding_object = DataEncoding(
            "/Volumes/SJSU/CS271 Topics in Machine Learning/Final project/Dataset/main_frame.pkl")
X,y = data_encoding_object.fetch_encoded_sampled_Data()
print(X[X.isnull().any(axis=1)])
unsupervised_learning = KMeans(n_clusters=2,random_state=0).fit(X)

print(unsupervised_learning.labels_)
print("Nothing")
