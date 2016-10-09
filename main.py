import pandas as pd
import numpy as np
import datetime
from sklearn.linear_model import LinearRegression

def load_data():
    train = pd.read_csv('./data/train.csv')
    features = pd.read_csv('./data/features.csv')
    test = pd.read_csv('./data/test.csv')
    
    features = clean_features(features)

    train = clean_train_markdown(train)
    return (train, test, features)
   
def clean_features(features):
    features = features[['Store', 'Date', 'Temperature', 'Fuel_Price', 'IsHoliday']]
    return features
 
    
def combine_features(train, test, features):
    train = np.array(train)
    test = np.array(test)
    features = np.array(features)
    X_train, y_train, X_test, dates = [], [], [], []

    for i in range(len(train)):
        X_train.append([])
        store, dept, date, sales, isHoliday = train[i]
        extra_features = find_features(store, date, features)
        y_train.append(sales)
        X_train[i] = list(extra_features)

        temp_date = date.split('-')
        year, month, day = int(temp_date[0]), int(temp_date[1]), int(temp_date[2])
        curr_date = datetime.date(year, month, day)
        one_year = datetime.timedelta(days=365)
        last_year = str(curr_date - one_year)
        week = datetime.timedelta(days=7)
        prev_week = curr_date - week
        prev_week = str(prev_week)
        next_week = curr_date + week
        next_week = str(next_week)
        prev_week = get_holidays(prev_week)
        curr_week = get_holidays(date)
        next_week = get_holidays(next_week)
        #last_years_sales = get_last_years_sales(curr_date, last_year, train)
        X_train[i] = X_train[i] + prev_week + curr_week + next_week

        
    for i in range(len(test)):
        X_test.append([])
        store, dept, date, isHoliday = test[i]
        extra_features = find_features(store, date, features)
        X_test[i] = list(extra_features)

        temp_date = date.split('-')
        year, month, day = int(temp_date[0]), int(temp_date[1]), int(temp_date[2])
        curr_date = datetime.date(year, month, day)
        week = datetime.timedelta(days=7)
        one_year = datetime.timedelta(days=365)
        last_year = str(curr_date - one_year)
        prev_week = str(curr_date - week)
        next_week = str(curr_date + week)
        prev_week = get_holidays(prev_week)
        curr_week = get_holidays(date)
        next_week = get_holidays(next_week)
        #last_years_sales = get_last_years_sales(curr_date, last_year, train)
        X_test[i] = X_test[i] + prev_week + curr_week + next_week
        dates.append(date)

    return(X_train, X_test, y_train, dates)


    
    
def get_holidays(date):
    super_bowl = ['2010-02-12','2011-02-11','2012-02-10','2013-02-08']
    labor_day = ['2010-09-10','2011-09-09','2012-09-07','2013-09-06']
    thanksgiving = ['2010-11-26','2011-11-25','2012-11-23','2013-11-29']
    christmas = ['2010-12-31','2011-12-30','2012-12-28','2013-12-27']
    if date in super_bowl:
        return [0,0,0,1]
    elif date in labor_day:
        return [0,0,1,0]
    elif date in thanksgiving:
        return [0,1,0,0]
    elif date in christmas:
        return [1,0,0,0]
    else:
        return [0,0,0,0]    

def find_features(store, date, features):
    for i in range(len(features)):
        if features[i][0] == store and features[i][1] == date:
            return features[i][2:-1]

def linear_model(X_train, y_train, X_test):
    clf = LinearRegression()
    clf.fit(X_train, y_train)
    y_test = clf.predict(X_test)
    return y_test



def write(y_test,store,dept,dates):
    f = open('./data/result.csv','a')
    for i in range(len(y_test)):
        Id = str(store)+'_'+str(dept)+'_'+str(dates[i])
        sales = y_test[i]
        f.write('%s,%s\n'%(Id,sales))
    f.close()
        
def main(): 
    f = open('./data/result.csv','w')
    f.write('Id,Weekly_Sales\n')
    f.close()
    train, test, features = load_data()
    print('Staring....This may take a while....')
    for i in range(1,46):
        store_train = train[train['Store'] == i]
        store_test = test[test['Store'] == i]
        store_features = features[features['Store'] == i]
        train_depts = list(set(store_train['Dept'].values))
        test_depts = list(set(store_test['Dept'].values))
        for dept in test_depts:
            if dept not in train_depts:
                print('dept %s, store: %s is missing training data'%(dept,i))
                tests = store_test[store_test['Dept'] == dept]
                dates = list(tests['Date'])
                y=[0 for j in range(len(tests))]
                write(y,i,dept,dates)
        for dept in train_depts:
            train_data = store_train[store_train['Dept'] == dept]
            test_data = store_test[store_test['Dept'] == dept]

            X_train, X_test, y_train, dates = combine_features(train_data, test_data, store_features)
            if len(X_test) > 0:
                y_test = linear_model(X_train, y_train, X_test)
                write(y_test, i , dept, dates)
main()