import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('MrBeast_youtube_stats.csv')

# Select only the required columns
df = df[['duration_seconds', 'viewCount', 'likeCount', 'commentCount']]

# Separate features and target
X = df.drop('viewCount', axis=1)
y = df['viewCount']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

# Recombine for preprocessing
train_data = X_train.copy()
train_data['viewCount'] = y_train
test_data = X_test.copy()
test_data['viewCount'] = y_test

# Reshape target variables
train_y = train_data['viewCount'].values.reshape((-1, 1))
test_y = test_data['viewCount'].values.reshape((-1, 1))

# Impute missing values in target if any
impy = SimpleImputer(strategy="mean")
impy.fit(train_y)
train_y = impy.transform(train_y)
test_y = impy.transform(test_y)

# Drop target variable from main data
train_data = train_data.drop(columns=['viewCount'])
test_data = test_data.drop(columns=['viewCount'])

# Create pipeline for processing numeric variables
numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]
)

# Create the full preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, make_column_selector(dtype_include=['int', 'float']))
    ]
)

# Create the complete pipeline
clf = Pipeline(
    steps=[("preprocessor", preprocessor)]
)

# Fit and transform the data
clf.fit(train_data, train_y)
train_new = clf.transform(train_data)
test_new = clf.transform(test_data)

# Convert to dataframe
train_new = pd.DataFrame(train_new, columns=X_train.columns)
test_new = pd.DataFrame(test_new, columns=X_test.columns)

# Add target back
train_new['viewCount'] = train_y
test_new['viewCount'] = test_y

# Save processed data
train_new.to_csv('data/processed_mrbeast_train.csv', index=False)
test_new.to_csv('data/processed_mrbeast_test.csv', index=False)

# Save pipeline
with open('data/mrbeast_pipeline.pkl', 'wb') as f:
    pickle.dump(clf, f)