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


class UnsupervisedLearning:
    def visualise_confusion_matrix(self, data):
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
        plt.show()

    def compare_models(self, accuracy_comparison, accuracy_table_cols):
        for algorithm in machine_learning_algorithms:
            algorithm.fit(X_train, y_train)
            algorithm_name = algorithm.__class__.__name__
            print(algorithm_name)
            y_pred = algorithm.predict(X_test)
            accuracy = metrics.accuracy_score(y_test, y_pred)
            print("Accuracy", accuracy)
            # cv_score = np.mean(cross_val_score(algorithm,X_train,y_train,cv=5))
            # print("Cross Validation Score",cv_score)
            print(classification_report(y_test, y_pred))
            confusion_matrix_values = confusion_matrix(y_test, y_pred)

            true_negative, false_positive, false_negative, true_positive = confusion_matrix_values.ravel()

            fpr_value = false_positive / (true_negative + false_positive)
            error_rate = 1 - accuracy
            precision_value = precision_score(y_test, y_pred)
            recall_value = recall_score(y_test, y_pred)
            auc_value = roc_auc_score(y_test, y_pred)

            print(precision_value)
            print(recall_value)

            accuracy_table = pd.DataFrame([[algorithm, round(accuracy * 100, 2), round(error_rate * 100, 2),
                                            round(fpr_value * 100, 2), round(auc_value * 100, 2)
                                               , round(precision_value * 100, 2), round(recall_value * 100, 2),
                                            round(69 * 100, 2)]],
                                          columns=accuracy_table_cols)
            accuracy_comparison = accuracy_comparison.append(accuracy_table)
            unsupervised_learning.visualise_confusion_matrix(confusion_matrix_values)

        print(accuracy_comparison.head())

    def other_models(self):
        svm_classifier = svm.SVC(kernel='linear')
        svm_classifier.fit(X_train, y_train)
        y_pred = svm_classifier.predict(X_test)
        print("Accuracy Report", classification_report(y_test, y_pred))


unsupervised_learning = UnsupervisedLearning()
data_encoding_object = DataEncoding(
    "/Volumes/SJSU/CS271 Topics in Machine Learning/Final project/Dataset/main_frame.pkl")
main_dataset, X, y = data_encoding_object.fetch_encoded_sampled_Data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=109)

machine_learning_algorithms = [
    RandomForestClassifier(n_estimators=500, min_samples_split=10, max_depth=70, criterion='entropy',
                           max_features=X.shape[1],
                           random_state=42),
    AdaBoostClassifier(n_estimators=500, learning_rate=0.05, random_state=42),
    KNeighborsClassifier(n_neighbors=17)]

accuracy_table_cols = ["Algorithm", "Accuracy", "Error", "FPR", "AUC", "Precision", "Recall", "Cross Validation Score"]
accuracy_comparison = pd.DataFrame(columns=accuracy_table_cols)
unsupervised_learning.compare_models(accuracy_comparison, accuracy_table_cols)

balanced_ensemble_algorithms = [
    BalancedRandomForestClassifier(n_estimators=500, sampling_strategy='majority', min_samples_split=10,
                                   min_samples_leaf=1, criterion='entropy', max_depth=70,
                                   max_features=X.shape[1],
                                   replacement=True,
                                   random_state=42),
    BalancedBaggingClassifier(n_estimators=500, max_features=X.shape[1], replacement=True,
                              sampling_strategy='majority', random_state=42)]

