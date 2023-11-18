import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

class A:

    def get_clean_data(self):
        df=pd.read_csv('data.csv')
        df = df.drop(['Unnamed: 32', 'id'], axis=1)
        df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})  # convert Malignan to 1 and benign to 0
        return df

    def create_model(self, data):
        X = data.drop(['diagnosis'], axis=1)
        Y = data['diagnosis']

        # scale the data
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # split the data
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        # train the data
        model = LogisticRegression()
        model.fit(X_train, Y_train)

        # test the model
        y_pred = model.predict(X_test)
        print('Accuracy of our model:', accuracy_score(Y_test, y_pred))
        print("classification report:\n", classification_report(Y_test, y_pred))
        return X,Y,y_pred,model, scaler

    def main(self):
        data = self.get_clean_data()
        print(data.info())

        model, scaler = self.create_model(data)
        with open('model/model.pkl', 'wb') as f:
            pickle.dump(model, f)
        with open('model/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)

if __name__ == '__main__':
    obj = A()
    obj.main()
