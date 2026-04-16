# ==============================
# 8. Feature Scaling (Optional but good)
# ==============================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==============================
# 9. Train Model
# ==============================
model = LinearRegression()
model.fit(X_train, y_train)