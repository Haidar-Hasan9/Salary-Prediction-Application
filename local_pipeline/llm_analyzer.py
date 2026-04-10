import pandas as pd
import matplotlib.pyplot as plt
import ollama
import io
import base64
from typing import List, Dict
import os
from dotenv import load_dotenv
import os
load_dotenv()  # loads .env file

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

def generate_predictions_df():
    """Simulate predictions for a representative set of job profiles."""
    data = [
        {"experience_level": "EN", "job_title": "Data Scientist", "work_year": 2023, "employee_residence": "US", "predicted_salary_usd": 72442},
        {"experience_level": "MI", "job_title": "Data Scientist", "work_year": 2023, "employee_residence": "US", "predicted_salary_usd": 96360},
        {"experience_level": "SE", "job_title": "Data Scientist", "work_year": 2023, "employee_residence": "US", "predicted_salary_usd": 139775},
        {"experience_level": "EX", "job_title": "Data Scientist", "work_year": 2023, "employee_residence": "US", "predicted_salary_usd": 176200},
        {"experience_level": "SE", "job_title": "Machine Learning Engineer", "work_year": 2023, "employee_residence": "US", "predicted_salary_usd": 139775},
        {"experience_level": "SE", "job_title": "Data Analyst", "work_year": 2023, "employee_residence": "US", "predicted_salary_usd": 139775},
        {"experience_level": "SE", "job_title": "Data Scientist", "work_year": 2023, "employee_residence": "IN", "predicted_salary_usd": 59773},
    ]
    return pd.DataFrame(data)

def generate_llm_analysis(df: pd.DataFrame):
    """Send data to Ollama and get narrative + chart code."""
    summary = df.groupby("experience_level")["predicted_salary_usd"].mean().round(2).to_dict()
    summary_text = ", ".join([f"{k}: ${v:,.0f}" for k, v in summary.items()])
    
    prompt = f"""
You are a data analyst. Given the following average salaries by experience level for data science jobs (in USD):
{summary_text}

Write a short narrative (2-3 paragraphs) that:
- Compares salaries across experience levels.
- Explains why senior roles earn more.
- Mentions how location (US vs IN) affects salary.
- Gives actionable advice for someone looking to increase their salary.

Then, generate Python matplotlib code (inside triple backticks) that creates a bar chart of average salary by experience level.
The code should be self-contained and use the data provided: 
experience_levels = ['EN','MI','SE','EX']
avg_salaries = [72442, 96360, 139775, 176200]

**Important:** The code must save the chart to a file named 'chart.png' using plt.savefig('chart.png'), and should NOT call plt.show().

Example:
import matplotlib.pyplot as plt
experience_levels = ['EN','MI','SE','EX']
avg_salaries = [72442, 96360, 139775, 176200]
plt.figure(figsize=(8,5))
plt.bar(experience_levels, avg_salaries)
plt.xlabel('Experience Level')
plt.ylabel('Average Salary (USD)')
plt.title('Average Salary by Experience Level')
plt.savefig('chart.png')
"""
    response = ollama.chat(model=OLLAMA_MODEL, messages=[{'role': 'user', 'content': prompt}])
    llm_output = response['message']['content']
    
    narrative = llm_output.split("```python")[0].strip()
    
    chart_code = ""
    if "```python" in llm_output:
        code_part = llm_output.split("```python")[1].split("```")[0]
        chart_code = code_part
    else:
        chart_code = """
import matplotlib.pyplot as plt
experience_levels = ['EN','MI','SE','EX']
avg_salaries = [72442, 96360, 139775, 176200]
plt.figure(figsize=(8,5))
plt.bar(experience_levels, avg_salaries, color='skyblue')
plt.xlabel('Experience Level')
plt.ylabel('Average Salary (USD)')
plt.title('Average Salary by Experience Level')
plt.savefig('chart.png')
"""
    return narrative, chart_code

def execute_chart_code(chart_code: str, output_image="chart.png"):
    """Execute the chart code and return base64 encoded image."""
    # Write code to a temporary file
    with open("temp_chart.py", "w") as f:
        f.write(chart_code)
    
    # Run the script
    import subprocess
    result = subprocess.run(["python", "temp_chart.py"], capture_output=True, text=True)
    
    if result.returncode != 0:
        print("Error executing chart code:")
        print(result.stderr)
        # Fallback: create a simple chart directly
        print("Creating fallback chart...")
        experience_levels = ['EN', 'MI', 'SE', 'EX']
        avg_salaries = [72442, 96360, 139775, 176200]
        plt.figure(figsize=(8,5))
        plt.bar(experience_levels, avg_salaries, color='skyblue')
        plt.xlabel('Experience Level')
        plt.ylabel('Average Salary (USD)')
        plt.title('Average Salary by Experience Level')
        plt.savefig(output_image)
    
    # Clean up temp file
    os.remove("temp_chart.py")
    
    # Convert image to base64
    with open(output_image, "rb") as img_file:
        img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
    return img_base64

if __name__ == "__main__":
    df = generate_predictions_df()
    print("Data for analysis:")
    print(df)
    
    narrative, chart_code = generate_llm_analysis(df)
    print("\n" + "="*60)
    print("LLM NARRATIVE:")
    print("="*60)
    print(narrative)
    
    print("\n" + "="*60)
    print("GENERATED CHART CODE:")
    print("="*60)
    print(chart_code)
    
    img_base64 = execute_chart_code(chart_code)
    print("\nChart saved as chart.png and base64 string generated (first 100 chars):")
    print(img_base64[:100] + "...")