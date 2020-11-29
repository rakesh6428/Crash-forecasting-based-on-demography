from sklearn.model_selection import train_test_split
from sklearn import svm
from Core.data_encoding_sampling import *
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from imblearn.ensemble import BalancedRandomForestClassifier, BalancedBaggingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA


class UnsupervisedLearning:
    def visualise_confusion_matrix(self, data, count):
        plt.figure(figsize=(10, 5))
        plt.clf()
        plt.imshow(data, interpolation='nearest', cmap='tab20')
        classNames = ['No Loyalty', 'Loyalty']
        plt.title('Confusion Matrix')
        plt.ylabel('Actual\n')
        plt.xlabel('Predicted\n')
        tick_marks = np.arange(len(classNames))
        plt.xticks(tick_marks, classNames)
        plt.yticks(tick_marks, classNames)
        s = [['TN', 'FP'], ['FN', 'TP']]
        for i in range(2):
            for j in range(2):
                plt.text(j, i, str(s[i][j]) + "=" + str(data[i][j]), horizontalalignment='center',
                         color='black')
        plt.savefig('Views/Model_Scoring_'+str(count)+'.png')
        plt.show()

    def compare_models(self, accuracy_comparison, accuracy_table_cols, X_train, y_train, machine_learning_algorithms):
        count = 0
        for algorithm in machine_learning_algorithms:
            algorithm.fit(X_train, y_train)
            algorithm_name = algorithm.__class__.__name__
            print(algorithm_name)
            y_pred = algorithm.predict(X_test)
            accuracy = metrics.accuracy_score(y_test, y_pred)
            print("\nAccuracy", accuracy)
            cv_score = np.mean(cross_val_score(algorithm,X_train,y_train,cv=5))
            print("Cross Validation Score",cv_score)
            print("\nAccuracy Report:\n",classification_report(y_test, y_pred))
            confusion_matrix_values = confusion_matrix(y_test, y_pred)

            true_negative, false_positive, false_negative, true_positive = confusion_matrix_values.ravel()

            fpr_value = false_positive / (true_negative + false_positive)
            error_rate = 1 - accuracy
            precision_value = precision_score(y_test, y_pred)
            recall_value = recall_score(y_test, y_pred)
            auc_value = roc_auc_score(y_test, y_pred)

            accuracy_table = pd.DataFrame([[algorithm_name, round(accuracy * 100, 2), round(error_rate * 100, 2),
                                            round(fpr_value * 100, 2), round(auc_value * 100, 2)
                                               , round(precision_value * 100, 2), round(recall_value * 100, 2),
                                            round(69 * 100, 2)]],
                                          columns=accuracy_table_cols)
            accuracy_comparison = accuracy_comparison.append(accuracy_table)

            unsupervised_learning.visualise_confusion_matrix(confusion_matrix_values,count)
            count += 1

        return accuracy_comparison

    def other_models(self):
        svm_classifier = svm.SVC(kernel='linear')
        svm_classifier.fit(X_train, y_train)
        y_pred = svm_classifier.predict(X_test)
        print("\nAccuracy Report", classification_report(y_test, y_pred))

    def visualise_all_model_comparison(self,accuracy_visualise):

        fig, ax = plt.subplots(nrows=7, ncols=1, figsize=(10, 20))
        sns.barplot(x='Accuracy', y='Algorithm', data=accuracy_visualise, palette='plasma', ax=ax[0])
        sns.barplot(x='Error', y='Algorithm', data=accuracy_visualise, palette='plasma', ax=ax[1])
        sns.barplot(x='FPR', y='Algorithm', data=accuracy_visualise, palette='plasma', ax=ax[2])
        sns.barplot(x='AUC', y='Algorithm', data=accuracy_visualise, palette='plasma', ax=ax[3])
        sns.barplot(x='Precision', y='Algorithm', data=accuracy_visualise, palette='plasma', ax=ax[4])
        sns.barplot(x='Recall', y='Algorithm', data=accuracy_visualise, palette='plasma', ax=ax[5])
        sns.barplot(x='Cross Validation Score', y='Algorithm', data=accuracy_visualise, palette='plasma', ax=ax[6])
        plt.tight_layout()
        plt.savefig('Views/Final Comparison.png')
        plt.show()

unsupervised_learning = UnsupervisedLearning()
data_encoding_object = DataEncoding(
    "/Volumes/SJSU/CS271 Topics in Machine Learning/Final project/Dataset/main_frame.pkl")
main_dataset, X, y, encoded_X, encoded_y = data_encoding_object.fetch_encoded_sampled_Data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=109)


machine_learning_algorithms = [
    RandomForestClassifier(n_estimators=500, min_samples_split=10, max_depth=70, criterion='entropy',
                           max_features=encoded_X.shape[1],
                           random_state=42),
    AdaBoostClassifier(n_estimators=500, learning_rate=0.05, random_state=42),
    KNeighborsClassifier(n_neighbors=17)
]

accuracy_table_cols = ["Algorithm", "Accuracy", "Error", "FPR", "AUC", "Precision", "Recall", "Cross Validation Score"]
accuracy_comparison = pd.DataFrame(columns=accuracy_table_cols)
accuracy_comparison_ensemble = unsupervised_learning.compare_models(accuracy_comparison, accuracy_table_cols, X_train, y_train, machine_learning_algorithms)

balanced_ensemble_algorithms = [
    BalancedRandomForestClassifier(n_estimators=500, sampling_strategy='majority', min_samples_split=10,
                                   min_samples_leaf=1, criterion='entropy', max_depth=35,
                                   max_features=encoded_X.shape[1],
                                   replacement=True,
                                   random_state=42),
    BalancedBaggingClassifier(n_estimators=500, max_features=encoded_X.shape[1], replacement=True,
                              sampling_strategy='majority', random_state=42)
]


accuracy_comparison_balanced = unsupervised_learning.compare_models(accuracy_comparison, accuracy_table_cols, X_train, y_train, balanced_ensemble_algorithms)
accuracy_visualise = pd.concat([accuracy_comparison_ensemble,accuracy_comparison_balanced])
unsupervised_learning.visualise_all_model_comparison(accuracy_visualise)

#------- Perform PCA ---------
pca = PCA()
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

print("\n PCA Results:")
print(X_test)
explained_variance = pca.explained_variance_ratio_
print(explained_variance)
