# Import Modules
import pandas as pd
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Hide all Warning messages
warnings.filterwarnings('ignore')


df = pd.read_csv('./dataset/wdbc.data')
df.columns = ['ID','Diagnosis','radius','texture','perimeter','area','smoothness','compactness','concavity','concave_points',
              'symmetry','fractal_dimension','radius_SE','texture_SE','perimeter_SE','area_SE','smoothness_SE','compactness_SE',
              'concavity_SE','concave_points_SE','symmetry_SE','fractal_dimension_SE','radius_Worst','texture_Worst','perimeter_Worst',
              'area_Worst','smoothness_Worst','compactness_Worst','concavity_Worst','concave_points_Worst','symmetry_Worst',
              'fractal_dimension_Worst']

# Splite Target and Features
target = df['Diagnosis']
features = df.drop(['ID','Diagnosis'],axis=1)
x_Features_Selected = features[['area','concavity','radius',]]

# split data into 80 % for trainning and 20% for testing
x_train, x_test, y_train, y_test = train_test_split(x_Features_Selected, target, test_size=0.2, random_state=42)


# Random Forest Classification Model
rfc_model = RandomForestClassifier(random_state=43)      
rfc_model = rfc_model.fit(x_train,y_train)

# Accuracy
Score = accuracy_score(y_test,rfc_model.predict(x_test))
print('\nAccuracy: {:.2f}%'.format(Score*100))