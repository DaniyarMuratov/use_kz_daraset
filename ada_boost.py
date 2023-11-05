import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

data = pd.read_csv('wfp_food_prices_kaz.csv')

data = data.drop(['date', 'market'], axis=1)

base_classifier = DecisionTreeClassifier(max_depth=5)
model = AdaBoostClassifier(base_classifier, n_estimators=100, random_state=64)
model.fit(data.drop('commodity', axis=1), data['commodity'])

predicted_clusters = model.predict(data.drop('commodity', axis=1))
data['cluster'] = predicted_clusters

accuracy = accuracy_score(data['commodity'], predicted_clusters)
print(f"Accuracy: {round(accuracy, 2)}")

cluster_counts = data['cluster'].value_counts().reset_index()
cluster_counts.columns = ['Cluster', 'Count']

plt.bar(["Картошка", "Мясо", "Масло", "Молоко", "Мука"], cluster_counts['Count'])
plt.xlabel('Классы')
plt.ylabel('Число')
plt.title('Число классов')
plt.show()
