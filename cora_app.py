# -*- coding: utf-8 -*-
"""
Created on Sat May 31 14:23:23 2025

@author: mansi bhat
"""
import streamlit as st
import pandas as pd
import openai

# Load customer data
df = pd.read_csv("dummy_customer_data.csv")

# Set your OpenAI key
openai.api_key = "sk-..."  # Replace with your actual key

# App title
st.title("🧠 CORA – Campaign Message Generator")

# Select strategy
strategy = st.selectbox("🎯 Choose a campaign strategy", ["upsell", "cross-sell", "retention"])

# Generate GPT message
def generate_message(row, strategy):
    if strategy == "upsell":
        prompt = f"Create an upsell offer for a {row['age']} y/o {row['gender']} who bought {row['product_bought']}."
    elif strategy == "cross-sell":
        prompt = f"Suggest a cross-sell product to a {row['age']} y/o {row['gender']} who bought {row['product_bought']}."
    else:
        prompt = f"Write a friendly message for a {row['age']} y/o {row['gender']} who hasn't bought in {row['last_purchase_days_ago']} days."

    try:
        client = openai.OpenAI(api_key=openai.api_key)
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

# Button to generate messages
if st.button("💬 Generate Marketing Messages"):
    with st.spinner("Generating..."):
        df['marketing_message'] = df.apply(lambda row: generate_message(row, strategy), axis=1)
    st.success("Done!")

# Display results
if 'marketing_message' in df.columns:
    st.dataframe(df[['customer_id', 'product_bought', 'marketing_message']])
    st.download_button("📥 Download as CSV", df.to_csv(index=False), file_name="cora_output.csv")

