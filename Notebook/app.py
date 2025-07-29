import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ollama
import os

# Create a folder for saving plots
os.makedirs("plots", exist_ok=True)

# Function to generate AI-powered insights using Mistral
def generate_ai_insights(df_summary):
    prompt = f"Analyze the dataset summary and provide insights:\n\n{df_summary}"
    response = ollama.chat(
        model="mistral",
        messages=[{"role": "user", "content": prompt}]
    )
    return response['message']['content']

# Function to create data visualizations
def generate_visualizations(df):
    plot_paths = []

    # Histograms for numeric columns
    for col in df.select_dtypes(include=['number']).columns:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col], bins=30, kde=True, color="blue")
        plt.title(f"Distribution of {col}")
        path = f"plots/{col}_distribution.png"
        plt.savefig(path)
        plot_paths.append(path)
        plt.close()

    # Correlation heatmap
    numeric_df = df.select_dtypes(include=['number'])
    if not numeric_df.empty:
        plt.figure(figsize=(8, 5))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title("Correlation Heatmap")
        path = "plots/correlation_heatmap.png"
        plt.savefig(path)
        plot_paths.append(path)
        plt.close()

    return plot_paths

# Main function: EDA + AI analysis
def eda_analysis(file_path):
    try:
        if not file_path.endswith('.csv'):
            return "❌ Please upload a valid CSV file.", []

        df = pd.read_csv(file_path)

        # Handle missing values
        for col in df.select_dtypes(include=['number']).columns:
            df[col].fillna(df[col].median(), inplace=True)
        for col in df.select_dtypes(include=['object']).columns:
            df[col].fillna(df[col].mode()[0], inplace=True)

        # Summary & missing values
        summary = df.describe(include='all').to_string()
        missing_values = df.isnull().sum().to_string()

        # AI-generated insights
        insights = generate_ai_insights(summary)

        # Visualizations
        plot_paths = generate_visualizations(df)

        return (
            f"\n✅ Data Loaded Successfully!\n\n📋 Summary:\n{summary}"
            f"\n\n🔍 Missing Values:\n{missing_values}"
            f"\n\n🤖 AI-Generated Insights:\n{insights}",
            plot_paths
        )

    except Exception as e:
        return f"❌ An error occurred: {str(e)}", []

# Gradio UI
demo = gr.Interface(
    fn=eda_analysis,
    inputs=gr.File(type="filepath", label="Upload CSV File"),
    outputs=[gr.Textbox(label="EDA Report"), gr.Gallery(label="Data Visualizations")],
    title="📊 LLM-Powered Exploratory Data Analysis (EDA)",
    description="Upload any dataset CSV file and get automated EDA insights with AI-powered analysis and visualizations."
)

# Launch app
demo.launch(share=True)
