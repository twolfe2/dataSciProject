import pandas as pd
import numpy
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss

df = pd.read_csv('./data/train.csv')

train, test = train_test_split(df, test_size = 0.2)

X_train = train.loc[:,['Store', 'IsHoliday', 'Date', 'Dept']]
Y_train = train['Weekly_Sales']

X_test = test.loc[:,['Store', 'IsHoliday', 'Dept', 'Date']]
Y_test = test['Weekly_Sales'] 

clf = RandomForestClassifier(n_estimators=25)
clf = clf.fit(X_train, Y_train)