# Demand-prediction-with-XGBoost-and-SHAP
Final Year Project
# Import necessary libraries
import pandas as pd
import sklearn
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, Lasso, LinearRegression, ElasticNet
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

try:
    # Load the dataset
    d2 = pd.read_csv(r'extended_fmcg_demand_forecasting(cleaned version).csv')  # Dataset D2
except FileNotFoundError:
    print("Error: Dataset file not found. Please check the file path.")
    exit(1)

# Step 2: Data Cleaning and Preparation

# 1. Resolve duplicate or ambiguous column names in D2
d2 = d2.rename(columns={
    'Product_Category.1': 'Product_Category_Code',
    'Store_Location.1': 'Store_Location_Code',
    'Price_pre': 'Price',
    'Sales_Volume_pre': 'Sales_Volume',
    'Price_post': 'New_Price',
    'Sales_Volume_post': 'New_Sales_Volume'
})

# 2. Check and handle missing values
d2.fillna({col: d2[col].mean() if d2[col].dtype != 'object' else d2[col].mode()[0] for col in d2.columns}, inplace=True)

# 3. Standardize Date format in D2
d2['Date'] = pd.to_datetime(d2['Date'], format='%d/%m/%Y')

# 4. Encode categorical columns and store mappings, then drop original columns
from sklearn.preprocessing import LabelEncoder

categorical_columns_d2 = d2.select_dtypes(include=['object']).columns
encoders = {}
category_mappings = {
    'Household': 0, 'Personal Care': 1, 'Dairy': 2, 'Snacks': 3, 'Beverages': 4
}
store_location_mappings = {
    'urban': 0, 'rural': 1, 'suburban': 2
}
weekday_mappings = {
    'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
    'Friday': 4, 'Saturday': 5, 'Sunday': 6
}

for col in categorical_columns_d2:
    encoders[col] = LabelEncoder()
    d2[col] = encoders[col].fit_transform(d2[col])
    if col == 'Product_Category':
        d2['Product_Category_Code'] = d2[col].map({v: k for k, v in category_mappings.items()}).map(category_mappings).fillna(d2[col])
        d2.drop(columns=['Product_Category'], inplace=True)
    if col == 'Store_Location':
        d2['Store_Location_Code'] = d2[col].map({v: k for k, v in store_location_mappings.items()}).map(store_location_mappings).fillna(d2[col])
        d2.drop(columns=['Store_Location'], inplace=True)
    if col == 'Weekday':
        d2[col] = d2[col].map({v: k for k, v in weekday_mappings.items()}).map(weekday_mappings).fillna(d2[col])

# 5. Feature Engineering: Create additional features in D2
d2['Price_Difference'] = d2['New_Price'] - d2['Price']

# 6. Extract Date-derived features (excluding Date_year)
d2['Date_month'] = d2['Date'].dt.month
d2['Date_day'] = d2['Date'].dt.day

# Inspect date-derived features
print("\nDate-Derived Features Overview:")
print("Unique Months:", d2['Date_month'].unique())
print("Unique Days:", d2['Date_day'].unique())
print("Variance of Date_month:", d2['Date_month'].var())
print("Variance of Date_day:", d2['Date_day'].var())
print("\nCorrelation with New_Sales_Volume:")
print(d2[['Date_month', 'Date_day', 'New_Sales_Volume']].corr())

# 7. Final Checks
print("\nCleaned Dataset D2 Overview:")
print(d2.head())
print("\nShape of Cleaned D2:", d2.shape)

# Objective: Evaluate Multiple Models for Comparison
# Remove duplicate columns if any
d2 = d2.loc[:, ~d2.columns.duplicated()]

# Drop the original 'Date' column
d2.drop(columns=['Date'], inplace=True)

# Define features (X) and target (y) - Predicting 'Sales_Volume_post'
X = d2.drop(columns=['New_Sales_Volume'])
y = d2['New_Sales_Volume']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models for comparison
models = [
    ('XGBRegressor', xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.1, reg_lambda=1.0, random_state=42)),
    ('RandomForestRegressor', RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42)),
    ('ExtraTreesRegressor', ExtraTreesRegressor(n_estimators=150, max_depth=10, min_samples_split=5, random_state=42)),
    ('Ridge', Ridge(alpha=10)),
    ('Lasso', Lasso(alpha=0.1)),
    ('LinearRegression', LinearRegression()),
    ('ElasticNet', ElasticNet(alpha=0.1, l1_ratio=0.5))
]

# Evaluate each model and store results
results = []
xgboost_model = None  # To store the XGBoost model for SHAP and scenario planning
for name, model in models:
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    train_rmse = sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    results.append({
        'Model': name,
        'Training RMSE': train_rmse,
        'Testing RMSE': test_rmse,
        'Training R²': train_r2,
        'Testing R²': test_r2
    })
    print(f"Model: {name}")
    print(f"Training RMSE: {train_rmse:.4f}, R²: {train_r2:.4f}")
    print(f"Testing RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}")
    print("-" * 60)
    # Store the XGBoost model for later use
    if name == 'XGBRegressor':
        xgboost_model = model

# Create a comparison table
comparison_df = pd.DataFrame(results)
print("\nModel Comparison Table:")
print(comparison_df)

# Scatter plot for XGBoost predictions
plt.figure(figsize=(8, 6))
plt.scatter(y_test, xgb_preds, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Sales Volume")
plt.ylabel("Predicted Sales Volume")
plt.title("XGBoost Model: Actual vs Predicted Sales Volume")
plt.tight_layout()
plt.show()

# SHAP Analysis (only for XGBoost)
if xgboost_model is not None:
    print("\nPerforming SHAP Analysis for XGBRegressor...")
    explainer = shap.TreeExplainer(xgboost_model)
    shap_values = explainer.shap_values(X)

    # SHAP Feature Importance (Mean Absolute SHAP Values)
    print("\nSHAP Feature Importance (Mean Absolute SHAP Values):")
    shap_importance = pd.DataFrame({
        'Feature': X.columns,
        'Mean_SHAP': np.abs(shap_values).mean(axis=0)
    })
    shap_importance = shap_importance.sort_values(by='Mean_SHAP', ascending=False)
    print(shap_importance)

    # SHAP Summary Plot
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.title("SHAP Summary Plot (Feature Importance)")
    plt.tight_layout()
    plt.show()

    # SHAP Summary Plot (Detailed with Distribution)
    shap.summary_plot(shap_values, X, show=False)
    plt.title("SHAP Summary Plot (Feature Impact Distribution)")
    plt.tight_layout()
    plt.show()

# Function: Explain SHAP in plain language
def explain_shap_in_plain_language(shap_values, feature_names, prediction, base_value):
    print("\nExplanation of Prediction in Plain Language:")
    print(f"Predicted Sales Volume: {prediction:.0f} units (Average Expected: {base_value:.0f} units)")
    print("Key Factors Affecting the Prediction:")
    sorted_impacts = sorted(zip(shap_values, feature_names), key=lambda x: abs(x[0]), reverse=True)[:3]
    for value, name in sorted_impacts:
        impact = "increases" if value > 0 else "decreases"
        print(f"- {name}: {impact} sales by about {abs(value):.0f} units")

# Function: Predict demand based on user input and show SHAP explanation (only for XGBoost)
def predict_demand_with_shap(model, X_train, feature_names, category_mappings, store_location_mappings, weekday_mappings):
    """
    Predict demand based on user-input features and display SHAP explanation with plain-language summary and plots.

    Parameters:
    - model: Trained XGBRegressor model
    - X_train: Training data for SHAP background
    - feature_names: List of feature names for user input
    - category_mappings: Dictionary mapping Product_Category names to codes
    - store_location_mappings: Dictionary mapping Store_Location names to codes
    - weekday_mappings: Dictionary mapping Weekday names to codes
    """
    while True:
        print("\nPlease provide the following information to predict sales volume:")

        # Define user-friendly prompts and expected inputs
        feature_prompts = {
            'Product_Category_Code': f"Product Category (choose from: {list(category_mappings.keys())}): ",
            'Price': "Original Price (e.g., 10.99 for $10.99, non-negative): ",
            'Sales_Volume': "Previous Sales Volume (e.g., 100 for 100 units, non-negative): ",
            'New_Price': "New Price after change (e.g., 12.99 for $12.99, non-negative): ",
            'Promotion': "Promotion Active (enter 'yes' or 'no'): ",
            'Store_Location_Code': f"Store Location (choose from: {list(store_location_mappings.keys())}): ",
            'Weekday': f"Day of the Week (choose from: {list(weekday_mappings.keys())}): ",
            'Supplier_Cost': "Supplier Cost (e.g., 5.00 for $5.00, non-negative): ",
            'Replenishment_Lead_Time': "Replenishment Lead Time (e.g., 5 for 5 days, non-negative): ",
            'Stock_Level': "Stock Level (e.g., 50 for 50 units, non-negative): ",
            'Date_month': "Month (1-12, e.g., 6 for June): ",
            'Date_day': "Day of Month (1-31, e.g., 15): "
        }

        user_input = {}
        try:
            for feature in feature_names:
                if feature == 'Price_Difference':
                    continue  # Will calculate later
                while True:
                    prompt = feature_prompts.get(feature, f"{feature} (numeric): ")
                    value = input(prompt).strip().lower()

                    try:
                        if feature == 'Product_Category_Code':
                            if value.capitalize() not in category_mappings:
                                print(f"Invalid category. Choose from: {list(category_mappings.keys())}")
                                continue
                            user_input[feature] = float(category_mappings[value.capitalize()])
                        elif feature == 'Store_Location_Code':
                            if value not in store_location_mappings:
                                print(f"Invalid location. Choose from: {list(store_location_mappings.keys())}")
                                continue
                            user_input[feature] = float(store_location_mappings[value])
                        elif feature == 'Promotion':
                            if value not in ['yes', 'no']:
                                print("Please enter 'yes' or 'no' for Promotion.")
                                continue
                            user_input[feature] = 1.0 if value == 'yes' else 0.0
                        elif feature == 'Weekday':
                            if value.capitalize() not in weekday_mappings:
                                print(f"Invalid day. Choose from: {list(weekday_mappings.keys())}")
                                continue
                            user_input[feature] = float(weekday_mappings[value.capitalize()])
                        else:
                            user_input[feature] = float(value)
                            if feature in ['Price', 'New_Price', 'Sales_Volume', 'Supplier_Cost', 'Replenishment_Lead_Time', 'Stock_Level'] and user_input[feature] < 0:
                                print(f"{feature} must be non-negative.")
                                continue
                            if feature == 'Date_month' and not (1 <= user_input[feature] <= 12):
                                print("Month must be between 1 and 12.")
                                continue
                            if feature == 'Date_day' and not (1 <= user_input[feature] <= 31):
                                print("Day must be between 1 and 31.")
                                continue
                        break
                    except ValueError:
                        print(f"Invalid input for {feature}. Please enter a valid value.")

            # Calculate Price_Difference
            user_input['Price_Difference'] = user_input['New_Price'] - user_input['Price']

            # Create DataFrame from user input
            user_df = pd.DataFrame([user_input], columns=feature_names)

            # Predict with the model
            pred = model.predict(user_df)

            print(f"\nPredicted Sales Volume: {pred[0]:.2f} units")

            # Compute SHAP values for the user input
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(user_df)

            # Calculate base value (average prediction)
            base_value = explainer.expected_value

            # Plot SHAP force plot for the prediction
            shap.force_plot(
                base_value=base_value,
                shap_values=shap_values[0],
                features=user_df,
                feature_names=feature_names,
                matplotlib=True,
                show=False
            )
            plt.title("SHAP Force Plot for Predicted Sales Volume")
            plt.tight_layout()
            plt.show()

            # Plain-language SHAP explanation
            explain_shap_in_plain_language(shap_values[0], feature_names, pred[0], base_value)

            # Bar chart of top feature impacts
            plt.figure(figsize=(8, 6))
            top_indices = np.argsort(np.abs(shap_values[0]))[-5:]
            plt.barh(np.array(feature_names)[top_indices], shap_values[0][top_indices],
                     color=['red' if x > 0 else 'blue' for x in shap_values[0][top_indices]])
            plt.xlabel("Impact on Prediction (SHAP Value)")
            plt.title("Top Features Influencing Predicted Sales Volume")
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            print("Please try again.")
            continue

        # Ask if user wants to make another prediction
        print("\nWould you like to make another prediction? (yes/no)")
        if input().strip().lower() != 'yes':
            break

# Run SHAP and scenario planning only for XGBoost
if xgboost_model is not None:
    print("\nInteractive Scenario Planning (XGBRegressor):")
    predict_demand_with_shap(xgboost_model, X_train, X.columns.tolist(), category_mappings, store_location_mappings, weekday_mappings)
