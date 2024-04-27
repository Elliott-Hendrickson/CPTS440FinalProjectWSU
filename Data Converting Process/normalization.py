from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd

file_path = 'train.csv'
transformed_file_path = 'transformed_data.csv'

data = pd.read_csv(file_path)

print(data.head(), data.dtypes)

# Select columns of numerical data
numeric_features = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
numeric_features.remove('Id')  # Remove the 'Id' column

# Select columns of categorical data
categorical_features = data.select_dtypes(include=['object']).columns.tolist()

# Scaling for numerical data
numeric_transformer = MinMaxScaler()

# Converting non numerical data to numerical data
categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse=False)

# Making preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Build pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

# Fit and transform data
transformed_data = pipeline.fit_transform(data)

# Get feature names
encoded_feature_names = (numeric_features +
                         list(pipeline.named_steps['preprocessor']
                              .named_transformers_['cat']
                              .get_feature_names_out(categorical_features)))

# New dataframe
transformed_df = pd.DataFrame(transformed_data, columns=encoded_feature_names)

print(transformed_df)
#transformed_df.to_csv(transformed_file_path, index=False)

# Calculate correlation matrix
correlation_matrix = transformed_df.corr()

print(correlation_matrix)
#correlation_matrix.to_csv('correlation_matrix.csv')