
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


file_path = 'index_1.csv'
try:
    df = pd.read_csv(file_path)
    print("✅ Data loaded successfully")
    print(df.head()) 
except FileNotFoundError:
    print(f"❌ File not found: {file_path}")


target_column = 'target' if 'target' in df.columns else None

if target_column:
    X = df.drop(columns=[target_column])
    y = df[target_column]
else:
    X = df.copy()
    y = None

# Drop identifier columns (like card/id)
identifier_cols = [col for col in X.columns if 'id' in col.lower() or 'card' in col.lower()]
id_tracking = X[identifier_cols].reset_index(drop=True) if identifier_cols else pd.DataFrame()
X = X.drop(columns=identifier_cols, errors='ignore')

# Convert date/datetime columns
if 'date' in X.columns:
    X['date'] = pd.to_datetime(X['date'], errors='coerce')
if 'datetime' in X.columns:
    X['datetime'] = pd.to_datetime(X['datetime'], errors='coerce')

# Drop high-cardinality object columns
high_card_cat_cols = [
    col for col in X.select_dtypes(include=['object', 'category']).columns
    if X[col].nunique() > 50
]
tracking_high_card = X[high_card_cat_cols].reset_index(drop=True) if high_card_cat_cols else pd.DataFrame()
X = X.drop(columns=high_card_cat_cols, errors='ignore')


numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # ✅ fixed: use `sparse_output` for latest sklearn
])

preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, numeric_cols),
    ('cat', categorical_pipeline, categorical_cols)
])


X_processed = preprocessor.fit_transform(X)
print("✅ Data transformed successfully")

# ✅ get feature names safely (works for sklearn >= 1.2)
cat_encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
cat_features = cat_encoder.get_feature_names_out(categorical_cols)
all_features = numeric_cols + list(cat_features)

# create DataFrame
processed_df = pd.DataFrame(X_processed, columns=all_features)

# Add target column back if present
if y is not None:
    processed_df['target'] = y.reset_index(drop=True)

# Add tracking columns back (if any)
if not id_tracking.empty:
    processed_df = pd.concat([id_tracking, processed_df], axis=1)
if not tracking_high_card.empty:
    processed_df = pd.concat([tracking_high_card, processed_df], axis=1)


output_file = 'cleaned_index_data.csv'
processed_df.to_csv(output_file, index=False)
print(f"✅ Transformed data saved to {output_file}")
print(processed_df.head())
