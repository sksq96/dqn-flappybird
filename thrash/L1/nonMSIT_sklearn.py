from six.moves import cPickle as pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

with open('train_test.pk', 'rb') as f:
	save = pickle.load(f)

Xr, yr = save['Xr'], save['yr']
Xe, ye = save['Xe'], save['ye']

clf = LogisticRegression()
clf.fit(Xr, yr)

expected = ye
predicted = clf.predict(Xe)

print(accuracy_score(expected, predicted))

