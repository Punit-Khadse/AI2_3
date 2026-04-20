import pandas as pd
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('../data/insurance.csv')

# Split features and target
X = df.drop('charges', axis=1)
y = df['charges']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Data split done")