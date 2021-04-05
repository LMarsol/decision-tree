import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def print_score(classifier, clf_data, clf_target):
    print(classifier.score(clf_data, clf_target))


def find_optimal_depth(min_value, max_value):
    # List of values to try for max_depth:
    max_depth_range = list(range(min_value, max_value))

    # List to store the average RMSE for each value of max_depth:
    accuracy = []

    for depth in max_depth_range:
        test_clf = DecisionTreeClassifier(max_depth=depth, random_state=0)
        test_clf.fit(X_train, y_train)

        test_score = test_clf.score(X_test, y_test)
        accuracy.append(test_score)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 7))

    ax.plot(max_depth_range,
            accuracy,
            lw=2,
            color='k')
    #
    ax.set_xlim([min_value, max_value - 1])
    ax.set_ylim([.50, 1.00])
    ax.grid(True, axis='both', zorder=0, linestyle=':', color='k')
    ax.set_xlabel('max_depth', fontsize=24)
    ax.set_ylabel('Accuracy', fontsize=24)
    fig.tight_layout()
    fig.savefig('images/max_depth_vs_accuracy.png', dpi=300)


data = load_iris()

df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
print(df.head())

X_train, X_test, y_train, y_test = train_test_split(df[data.feature_names], df['target'], random_state=0)

clf = DecisionTreeClassifier(max_depth=2, random_state=0)
clf.fit(X_train, y_train)

clf.predict(X_test.iloc[0].values.reshape(1, -1))
clf.predict(X_test[0:10])

print_score(clf, X_test, y_test)

find_optimal_depth(1, 6)

