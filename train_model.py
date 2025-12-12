import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
import joblib

# 1️⃣ Load the dataset
df = pd.read_csv("Sleep_Duration.csv")

# 2️⃣ Convert Bed_time into numeric features
df[['Bed_hour','Bed_minute']] = df['Bed_time'].str.split(':', expand=True).astype(int)
df = df.drop(columns=['Bed_time'])

# 3️⃣ Features & Targets
X = df.drop(columns=['Mem_Acc','Reaction'])
y = df[['Mem_Acc','Reaction']]

# 4️⃣ Encode targets
le_mem = LabelEncoder()
le_react = LabelEncoder()

y_enc = pd.DataFrame({
    'Mem_Acc': le_mem.fit_transform(y['Mem_Acc']),
    'Reaction': le_react.fit_transform(y['Reaction'])
})

# 5️⃣ Categorical and numerical columns
cat_cols = ['Room_*C','Sound','Light']
num_cols = [c for c in X.columns if c not in cat_cols]

# 6️⃣ Preprocessing + Model Pipeline
preprocess = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
    ('num', 'passthrough', num_cols)
])

clf = MultiOutputClassifier(RandomForestClassifier(n_estimators=200, random_state=42))

pipe = Pipeline([
    ('prep', preprocess),
    ('model', clf)
])

# 7️⃣ Train model
pipe.fit(X, y_enc)

# 8️⃣ Save model and encoders
joblib.dump(pipe, "model.pkl")
joblib.dump({'mem': le_mem, 'react': le_react}, "encoder.pkl")

print("✅ model.pkl and encoder.pkl generated successfully!")

