# Mini-Project for Fundamentals of Machine Learning Course
![background](./materials/ai_wp.jpg)
This repository contains the code and data for a mini-project on facial expression recognition using machine learning algorithms.

## 📑 Project Policy
- Team: group should consist of 3-4 students.

    |No.| Student Name               | Student ID |
    | --| ---------------------------| ---------- |
    | 1 | Trần Anh Quân              | 21110374   |
    | 2 | Trần Trọng Phúc            | 21110372   |
    | 3 | Trần Xuân Thắng            | 21110390   |
    | 4 | Tưởng Hoàng Ngọc Tuyền     | 21110444   |

- The submission deadline is strict: **11:59 PM** on **June 22nd, 2024**. Commits pushed after this deadline will not be considered.

## 📦 Project Structure

The repository is organized into the following directories:

- **/data**: This directory contains the facial expression dataset. You'll need to download the dataset and place it here before running the notebooks. (Download link provided below)
- **/notebooks**: This directory contains the Jupyter notebook ```EDA.ipynb```. This notebook guides you through exploratory data analysis (EDA) and classification tasks.

## ⚙️ Usage

This project is designed to be completed in the following steps:

1. **Fork the Project**: Click on the ```Fork``` button on the top right corner of this repository, this will create a copy of the repository in your own GitHub account. Complete the table at the top by entering your team member names.

2. **Download the Dataset**: Download the facial expression dataset from the following [link](https://mega.nz/file/foM2wDaa#GPGyspdUB2WV-fATL-ZvYj3i4FqgbVKyct413gxg3rE) and place it in the **/data** directory:

3. **Complete the Tasks**: Open the ```notebooks/EDA.ipynb``` notebook in your Jupyter Notebook environment. The notebook is designed to guide you through various tasks, including:
    
    1. Prerequisite
    2. Principle Component Analysis
    3. Image Classification
    4. Evaluating Classification Performance 
To include this table and the conclusions in your README on GitHub, you can format the table in Markdown. Here is the Markdown version of your table and the conclusions:

### Summary of Model Performance

| Model                             | Accuracy | Macro Avg Precision | Macro Avg Recall | Macro Avg F1-Score | Weighted Avg Precision | Weighted Avg Recall | Weighted Avg F1-Score |
|-----------------------------------|----------|----------------------|------------------|---------------------|------------------------|---------------------|------------------------|
| Decision Tree (Original)          | 0.3165   | 0.32                 | 0.26             | 0.26                | 0.32                   | 0.32                | 0.30                   |
| Decision Tree (PCA)               | 0.3034   | 0.30                 | 0.25             | 0.25                | 0.29                   | 0.30                | 0.29                   |
| Logistic Regression (Original)    | 0.3717   | 0.44                 | 0.30             | 0.30                | 0.36                   | 0.37                | 0.35                   |
| Logistic Regression (PCA)         | 0.3717   | 0.30                 | 0.30             | 0.28                | 0.35                   | 0.37                | 0.34                   |
| K-Nearest Neighbors (Original)    | 0.3502   | 0.33                 | 0.32             | 0.32                | 0.35                   | 0.35                | 0.35                   |
| K-Nearest Neighbors (PCA)         | 0.3611   | 0.34                 | 0.33             | 0.33                | 0.36                   | 0.36                | 0.35                   |
| Multilayer Perceptron (Original)  | 0.4249   | 0.41                 | 0.40             | 0.41                | 0.42                   | 0.42                | 0.42                   |
| Multilayer Perceptron (PCA)       | 0.4093   | 0.38                 | 0.39             | 0.38                | 0.41                   | 0.41                | 0.41                   |

### So sánh hiệu suất các mô hình phân loại khác nhau

Chúng ta sẽ so sánh hiệu suất của các mô hình phân loại khác nhau sử dụng các số liệu: độ chính xác, độ chính xác, khả năng thu hồi và điểm F1. Mục tiêu là xác định mô hình nào hoạt động tốt nhất và loại cảm xúc nào được dự đoán chính xác nhất và loại cảm xúc nào mô hình mắc nhiều lỗi nhất.

#### Độ chính xác (Accuracy)
Độ chính xác đo lường độ đúng đắn tổng thể của mô hình. Dưới đây là độ chính xác của từng mô hình:
- Decision Tree (Dữ liệu gốc): 0.3165
- Decision Tree (Dữ liệu PCA): 0.3034
- Logistic Regression (Dữ liệu gốc): 0.3717
- Logistic Regression (Dữ liệu PCA): 0.3717
- K-Nearest Neighbors (Dữ liệu gốc): 0.3502
- K-Nearest Neighbors (Dữ liệu PCA): 0.3611
- Multilayer Perceptron (Dữ liệu gốc): 0.4249
- Multilayer Perceptron (Dữ liệu PCA): 0.4093

Mô hình Multilayer Perceptron (MLP) trên dữ liệu gốc có độ chính xác cao nhất (0.4249), cho thấy nó dự đoán đúng nhãn nhiều hơn so với các mô hình khác.

#### Độ chính xác (Precision)
Độ chính xác đo lường bao nhiêu trong số các trường hợp được dự đoán là dương tính thực sự là dương tính. Dưới đây là độ chính xác trung bình của các mô hình:
- Decision Tree (Dữ liệu gốc): 0.32
- Decision Tree (Dữ liệu PCA): 0.30
- Logistic Regression (Dữ liệu gốc): 0.44
- Logistic Regression (Dữ liệu PCA): 0.30
- K-Nearest Neighbors (Dữ liệu gốc): 0.33
- K-Nearest Neighbors (Dữ liệu PCA): 0.34
- Multilayer Perceptron (Dữ liệu gốc): 0.41
- Multilayer Perceptron (Dữ liệu PCA): 0.38

Mô hình Logistic Regression trên dữ liệu gốc có độ chính xác trung bình cao nhất (0.44), cho thấy nó dự đoán các trường hợp dương tính chính xác hơn.

#### Thu hồi (Recall)
Khả năng thu hồi đo lường bao nhiêu trong số các trường hợp dương tính thực sự được mô hình phát hiện. Dưới đây là khả năng thu hồi trung bình của các mô hình:
- Decision Tree (Dữ liệu gốc): 0.26
- Decision Tree (Dữ liệu PCA): 0.25
- Logistic Regression (Dữ liệu gốc): 0.30
- Logistic Regression (Dữ liệu PCA): 0.30
- K-Nearest Neighbors (Dữ liệu gốc): 0.32
- K-Nearest Neighbors (Dữ liệu PCA): 0.33
- Multilayer Perceptron (Dữ liệu gốc): 0.40
- Multilayer Perceptron (Dữ liệu PCA): 0.39

Mô hình Multilayer Perceptron trên dữ liệu gốc có khả năng thu hồi trung bình cao nhất (0.40), cho thấy nó phát hiện được nhiều trường hợp dương tính hơn.

#### F1 (F1-Score)
F1- score là trung bình điều hòa giữa độ chính xác và khả năng thu hồi. Dưới đây là điểm F1 trung bình của các mô hình:
- Decision Tree (Dữ liệu gốc): 0.26
- Decision Tree (Dữ liệu PCA): 0.25
- Logistic Regression (Dữ liệu gốc): 0.30
- Logistic Regression (Dữ liệu PCA): 0.28
- K-Nearest Neighbors (Dữ liệu gốc): 0.32
- K-Nearest Neighbors (Dữ liệu PCA): 0.33
- Multilayer Perceptron (Dữ liệu gốc): 0.41
- Multilayer Perceptron (Dữ liệu PCA): 0.38

Mô hình Multilayer Perceptron trên dữ liệu gốc có điểm F1 trung bình cao nhất (0.41), cho thấy sự cân bằng tốt nhất giữa độ chính xác và khả năng thu hồi.

#### Nhận xét từ Confusion Matrix
Dưới đây là nhận xét từ các confusion matrix của từng mô hình:

**Decision Tree:**
- Dữ liệu gốc: Mô hình này hoạt động tốt nhất cho nhãn 3 (recall 0.53) nhưng mắc lỗi nhiều nhất cho nhãn 1 (recall 0.02).
- Dữ liệu PCA: Mô hình này hoạt động tốt nhất cho nhãn 3 (recall 0.59) nhưng mắc lỗi nhiều nhất cho nhãn 1 (recall 0.06).

**Logistic Regression:**
- Dữ liệu gốc: Mô hình này hoạt động tốt nhất cho nhãn 3 (recall 0.67) nhưng mắc lỗi nhiều nhất cho nhãn 1 (recall 0.02).
- Dữ liệu PCA: Mô hình này hoạt động tốt nhất cho nhãn 3 (recall 0.70) nhưng mắc lỗi nhiều nhất cho nhãn 1 (recall 0.00).

**K-Nearest Neighbors:**
- Dữ liệu gốc: Mô hình này hoạt động tốt nhất cho nhãn 3 (recall 0.51) nhưng mắc lỗi nhiều nhất cho nhãn 1 (recall 0.23).
- Dữ liệu PCA: Mô hình này hoạt động tốt nhất cho nhãn 3 (recall 0.56) nhưng mắc lỗi nhiều nhất cho nhãn 1 (recall 0.25).

**Multilayer Perceptron:**
- Dữ liệu gốc: Mô hình này hoạt động tốt nhất cho nhãn 3 (recall 0.58) và 5 (recall 0.54), nhưng mắc lỗi ít nhất cho nhãn 1 (recall 0.35).
- Dữ liệu PCA: Mô hình này hoạt động tốt nhất cho nhãn 3 (recall 0.60) nhưng mắc lỗi ít nhất cho
    Make sure to run all the code cells in the ```EDA.ipynb``` notebook and ensure they produce output before committing and pushing your changes.

5. **Commit and Push Your Changes**: Once you've completed the tasks outlined in the notebook, commit your changes to your local repository and push them to your forked repository on GitHub.


Feel free to modify and extend the notebook to explore further aspects of the data and experiment with different algorithms. Good luck.
