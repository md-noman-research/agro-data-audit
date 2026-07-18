# 🌱 Agro-Data-Audit (v0.2.3)

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/md-noman-research/agro-data-audit)
[![PyPI](https://img.shields.io/pypi/v/agro-data-audit.svg)](https://pypi.org/project/agro-data-audit/)

Welcome to the **agro-data-audit** framework! This package is designed to make data auditing, cleaning, Data Profiling (EDA), and Explainable Machine Learning (XAI) for agricultural datasets incredibly simple, robust, and professional.

*(**agro-data-audit** ফ্রেমওয়ার্কে স্বাগতম! এই প্যাকেজটি তৈরি করা হয়েছে এগ্রিকালচারাল ডেটাসেটগুলোর অডিটিং, ক্লিনিং, এক্সপ্লোরেটরি ডেটা অ্যানালাইসিস (EDA) এবং এক্সপ্লেইনেবল মেশিন লার্নিং (XAI)-কে সহজ ও প্রফেশনাল করার জন্য।)*

---

## 🏆 Test Results & Reliability
We take data integrity seriously. This package is fully tested:
- **Unit & Integration Tests**: ✅ Passed (27/27 tests, 100% success rate)
- **Performance Benchmark**: ⚡ Faster than manual pandas operations (`df.audit.scan()` vs native manual checks)
- **Real Dataset Tests**: ✅ Passed (Handles missing, duplicate, and extreme bounds seamlessly)
- **Code Coverage**: 🛡️ 87%+ Code Coverage

*(**টেস্টিং ও নির্ভরযোগ্যতা**: প্যাকেজটির শতভাগ টেস্ট (২৭/২৭) সফলভাবে পাশ করেছে এবং এটি সাধারণ ম্যানুয়াল প্যান্ডাস কোডের চেয়েও অনেক দ্রুত কাজ করে!)*

---

## 📥 1. Installation & Setup
Install the package via PyPI and import it alongside pandas.
*(প্যাকেজটি PyPI থেকে ইন্সটল করুন এবং প্যান্ডাসের সাথে ইম্পোর্ট করুন।)*

```python
# 1. Basic Installation (Core features)
pip install agro-data-audit

# 2. Full Installation (Includes Data Profiling)
pip install agro-data-audit[profile]
```

```python
# In your Python script / Jupyter Notebook
import pandas as pd
import data_audit  # Registers the 'audit' accessor automatically
```

---

## 🔍 2. Scanning Data for Issues (`scan`)
The first step is to scan the dataset for missing values, duplicates, and outliers. This method returns a clean `pd.DataFrame` containing all identified issues.
*(প্রথম ধাপে ডেটাসেটের মিসিং ভ্যালু, ডুপ্লিকেট এবং আউটলায়ার স্ক্যান করা হয়। এটি একটি পরিষ্কার DataFrame রিটার্ন করে।)*

### Basic Usage
```python
df = pd.read_csv("agricultural_data.csv")
issues = df.audit.scan()

if issues.empty:
    print("Dataset is perfectly clean!")
else:
    print(issues)
```

### Advanced Customization
You can control exactly how outliers are detected based on the nature of your agricultural data.
*(আপনার ডেটার ধরন অনুযায়ী আউটলায়ার ডিটেকশনের নিয়মগুলো কাস্টমাইজ করতে পারেন।)*

```python
# Use specific business logic boundaries (e.g., pH levels must strictly be between 5.0 and 8.5)
df.audit.scan(custom_bounds={
    'Soil_pH': (5.0, 8.5),
    'Temperature_C': (-10.0, 50.0)
})
```

---

## 🛠️ 3. Fixing the Data (`fix`)
Once the data is scanned, users can fix the issues. There are three powerful modes to suit every need:
*(ডেটা স্ক্যান করার পর আপনি সেটি ফিক্স করতে পারেন। এর ৩টি ভিন্ন মোড রয়েছে:)*

### A. Auto Mode (Fastest)
Automatically fills missing values (median for numbers, mode for text) and clips outliers to safe bounds.
*(স্বয়ংক্রিয়ভাবে সব মিসিং ভ্যালু এবং আউটলায়ার ফিক্স করে ফেলে।)*
```python
df.audit.fix(mode='auto')
```

### B. Suggest Mode (Consultative)
Doesn't change the data at all. Instead, it acts as an advisor, telling you exactly what should be done.
*(এটি ডেটা পরিবর্তন না করে, শুধু পরামর্শ দেয় যে কী করা উচিত।)*
```python
suggestions = df.audit.fix(mode='suggest')
```

### C. Manual Mode (Professional & Granular)
Allows you to precisely fix data by using the `sid` (Specific ID) helper.
*(নির্দিষ্ট রো বা কলাম নির্ভুলভাবে ফিক্স করার জন্য এটি ব্যবহার করা হয়।)*
```python
from data_audit import sid

df.audit.fix(
    mode="manual",
    fixes=[
        sid(10, 120.0),             # Fix Issue ID 10
        sid("Soil_Type", "Loamy"),  # Fill missing values in a column
        sid((15, "Age"), 99.0)      # Target specific cell (row 15, col 'Age')
    ]
)
```

---

## 📈 4. Data Profiling & EDA (`profile`)
Generate beautiful, comprehensive HTML Exploratory Data Analysis (EDA) reports instantly. 
*(এক ক্লিকেই সুন্দর HTML ডেটা প্রোফাইলিং রিপোর্ট জেনারেট করুন।)*

```python
# Generates "audit_report.html" in your folder
df.audit.profile()

# Advanced: Custom title, minimal mode, and large-dataset sampling
df.audit.profile(title="Crop Yield EDA", minimal=True, sample=50000)
```

---

## 📊 5. Generating Reports (`summary` & `report`)
Users can generate detailed statistical summaries and text-based audit reports.
*(ডেটার বিস্তারিত স্ট্যাটিস্টিকাল সামারি এবং টেক্সট রিপোর্ট দেখা যায়।)*

```python
print(df.audit.summary())
print(df.audit.report())
```

---

## 🤖 6. Machine Learning & AI (`ml`)
The package has an integrated Machine Learning module that automatically configures pipelines and models based on your data type.
*(এই প্যাকেজে একটি বিল্ট-ইন মেশিন লার্নিং মডিউল আছে যা ডেটার ধরন বুঝে মডেল তৈরি করে।)*

### Step 1: Recommend & Train a Model
*(মডেল রেকমেন্ড এবং ট্রেইন করা)*
```python
print(df.audit.ml.recommend(target='Crop_Yield'))
df.audit.ml.train(target='Crop_Yield')
```

### Step 2: Evaluate & Explain Predictions (XAI)
*(মডেল ইভ্যালুয়েট করা এবং প্রেডিকশনের ব্যাখ্যা জানা)*
```python
print(df.audit.ml.evaluate())

# Explain *why* the model made a specific prediction using SHAP
local_farm_data = df.iloc[0:1]
explanation = df.audit.ml.explain(local_data=local_farm_data)
print(explanation)
```

---

## 🚨 7. Finding Anomalies (`anomaly`)
Find completely weird or unusual rows in the dataset using Unsupervised Machine Learning (Isolation Forests).
*(পুরো ডেটাসেটের মধ্যে অস্বাভাবিক ডেটাগুলো বা অ্যানোমালি খুঁজে বের করা যায়।)*

```python
weird_farms = df.audit.anomaly()
print(weird_farms)
```

---

## 📜 8. Viewing History Log (`history`)
Transparency is key. Users can see a trail of every action the auditor performed on their data.
*(ডেটার উপর কী কী কাজ করা হয়েছে তার পুরো হিস্ট্রি লগ আকারে দেখা যায়।)*
```python
print(df.audit.history())
```

---

## 📜 License
This project is licensed under the strict open-source **AGPL-3.0 License**.
