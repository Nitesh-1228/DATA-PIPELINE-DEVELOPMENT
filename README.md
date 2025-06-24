# DATA-PIPELINE-DEVELOPMENT
COMPANY: CODTECH IT SOLUTIONS

NAME: IPPILI NITESH

INTERN ID: CT06DL1202

DOMAIN: DATA SCIENCE

DURATION: 6 WEEKS

MENTOR: NEELA SANTHOSH

# DESCRIPTION

This task demonstrates the creation of a complete ETL (Extract, Transform, Load) pipeline on the dataset index_1.csv. It automates all key preprocessing stages using Pandas, NumPy, and Scikit-learn, producing a clean dataset ready for machine learning workflows.
teps Performed

Step 1: Load the Data
The dataset index_1.csv is loaded using Pandas into a DataFrame. This allows us to analyze the raw data structure and check for missing values or inconsistencies.

Step 2: Separate Target Variable
We check if a column named target exists. If present, it is separated from the feature set so that we can preprocess features independently and later use target for modeling.

Step 3: Drop Identifier Columns
Columns containing terms like id or card (usually unique identifiers) are removed. These columns do not help in learning patterns and may lead to overfitting.

Step 4: Convert Date Columns
If the dataset contains date or datetime columns, they are converted into standard datetime format using pd.to_datetime() to make them analyzable or extract time-based features.

Step 5: Remove High-Cardinality Columns
Categorical columns with too many unique values (more than 50) are dropped, as one-hot encoding them would unnecessarily increase feature space and reduce model performance.

Step 6: Handle Missing Values
Missing numeric values are filled using the column mean, and missing categorical values are filled using the most frequent value, ensuring the dataset remains usable without dropping rows.

Step 7: Encode and Scale Features
Categorical features are encoded using OneHotEncoder and numeric features are standardized using StandardScaler to bring them onto the same scale, which improves model accuracy.

Step 8: Build and Apply Pipeline
Two pipelines (numeric and categorical) are built and combined using ColumnTransformer. This modular structure ensures cleaner, reusable, and scalable preprocessing logic.

Step 9: Reconstruct Final Dataset
The transformed features are combined into a new DataFrame. If present, target, identifier, and high-cardinality columns are added back to retain context.

Step 10: Saving Processed Data The final preprocessed training and testing sets are saved as CSV files:

index_1.csv

cleaned_index_data.csv These files can be used directly for training and validating machine learning models.

Outcome This script demonstrates a fully functional data preprocessing pipeline, automating the essential steps from raw data to clean, ready-to-use inputs. It is efficient, well-documented, and aligned with real-world data science practices.

# OUTPUT

https://github.com/user-attachments/assets/a9

