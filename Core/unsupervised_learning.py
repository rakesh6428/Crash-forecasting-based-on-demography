from Core.data_encoding_sampling import *
from sklearn.cluster import KMeans
from kmodes.kmodes import KModes
from sklearn.metrics import confusion_matrix

class UnsupervisedLearning:
    def create_cluster_data(self):
        clustered_dataset = main_dataset.copy().reset_index()
        accident_severity_labels = pd.DataFrame(severity_prediction)
        accident_severity_labels.columns = ['Cluster_Predicted']
        cluster_prediction = pd.concat([clustered_dataset, accident_severity_labels], axis=1).reset_index()
        cluster_prediction = cluster_prediction.drop(['index'], axis=1)
        return cluster_prediction
        print(cluster_prediction.head())

    def create_plots(self,cluster_prediction):
        fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(15, 12))
        sns.countplot(x=cluster_prediction['Speed_limit'],
                      order=cluster_prediction['Speed_limit'].value_counts().index,
                      hue=cluster_prediction['Cluster_Predicted'], palette='plasma', ax=ax[0])
        sns.countplot(x=cluster_prediction['Urban_or_Rural_Area'],
                      order=cluster_prediction['Urban_or_Rural_Area'].value_counts().index,
                      hue=cluster_prediction['Cluster_Predicted'], palette='plasma', ax=ax[1])
        sns.countplot(x=cluster_prediction['Junction_Detail'],
                      order=cluster_prediction['Junction_Detail'].value_counts().index,
                      hue=cluster_prediction['Cluster_Predicted'], palette='plasma', ax=ax[2])
        plt.tight_layout()
        plt.show()

    def create_actual_data_plots(self,unencoded_dataset):
        fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(15, 12))
        sns.countplot(x=unencoded_dataset['Speed_limit'],
                      order=unencoded_dataset['Speed_limit'].value_counts().index,
                      hue=unencoded_dataset['Serious_Class'], palette='plasma', ax=ax[0])
        sns.countplot(x=unencoded_dataset['Urban_or_Rural_Area'],
                      order=unencoded_dataset['Urban_or_Rural_Area'].value_counts().index,
                      hue=unencoded_dataset['Serious_Class'], palette='plasma', ax=ax[1])
        sns.countplot(x=unencoded_dataset['Junction_Detail'],
                      order=unencoded_dataset['Junction_Detail'].value_counts().index,
                      hue=unencoded_dataset['Serious_Class'], palette='plasma', ax=ax[2])
        plt.tight_layout()
        plt.show()


data_encoding_object = DataEncoding(
            "/Volumes/SJSU/CS271 Topics in Machine Learning/Final project/Dataset/main_frame.pkl")
main_dataset,X,y = data_encoding_object.fetch_encoded_sampled_Data()
unencoded_dataset = data_encoding_object.main_dataset
# print(X[X.isnull().any(axis=1)])

unsupervised_learning = KMeans(n_clusters=2,random_state=0)
severity_prediction = unsupervised_learning.fit_predict(main_dataset)


unsupervised_learning_KModes = KModes(n_clusters=2, init = "Huang", n_init = 1)
severity_prediction_KModes = unsupervised_learning_KModes.fit_predict(main_dataset)


unsupervised_learning = UnsupervisedLearning()
cluster_prediction=unsupervised_learning.create_cluster_data()

unsupervised_learning.create_plots(cluster_prediction)
unsupervised_learning.create_plots(cluster_prediction)




""""
print(severity_prediction)
cm = confusion_matrix(y,severity_prediction)
cm_KMode = confusion_matrix(y,severity_prediction_KModes)

ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
ax1 = sns.heatmap(cm_KMode, annot=True, fmt="d", cmap="Blues")
"""