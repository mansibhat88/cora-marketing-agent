
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import openai

#Load Data
df = pd.read_csv("dummy_customer_data.csv")
print(df.head())

#Segemntation using K-means

# Select features
features = df[['age', 'purchase_frequency', 'last_purchase_days_ago', 'total_spent']]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# Apply KMeans
kmeans = KMeans(n_clusters=3, random_state=0)
df['segment'] = kmeans.fit_predict(X_scaled)

print(df[['customer_id', 'segment']])

# Propensity Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X = features
y = df['campaign_response']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

df['propensity_score'] = model.predict_proba(X)[:, 1]

print(df[['customer_id', 'propensity_score']])



#Recommendation

openai.api_key = ""
# Choose one: "upsell", "cross-sell", or "retention"
target_strategy = "upsell"

# Function to generate marketing message based on strategy
def generate_message(row, strategy):
    if strategy == "upsell":
        prompt = (
            f"Create an upselling marketing message for a {row['age']}-year-old {row['gender']} "
            f"who bought {row['product_bought']}. Promote a premium or upgraded version. "
            f"Keep it short, persuasive, and product-focused."
        )
    elif strategy == "cross-sell":
        prompt = (
            f"Create a cross-selling message for a {row['age']}-year-old {row['gender']} "
            f"who bought {row['product_bought']}. Recommend a complementary product. "
            f"Make it feel helpful and personal."
        )
    elif strategy == "retention":
        prompt = (
            f"Create a retention message for a {row['age']}-year-old {row['gender']} "
            f"who hasn't purchased in {row['last_purchase_days_ago']} days. "
            f"Re-engage them with a friendly, emotional tone and subtle urgency."
        )
    else:
        return "Invalid strategy"

    try:
        client = openai.OpenAI(api_key=openai.api_key)
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error for customer {row['customer_id']}: {e}")
        return "Error generating message"
    

# Apply the function with the selected strategy
df['marketing_message'] = df.apply(lambda row: generate_message(row, target_strategy), axis=1)

# Save the final results
df.to_csv("cora_marketing_output.csv", index=False)
print(df[['customer_id', 'marketing_message']])

# Initialize OpenAI client

# Function to generate personalized marketing message

streamlit hello
