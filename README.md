# 📊 Ultra-Visual Data Analysis

![Build Status](https://img.shields.io/github/actions/workflow/status/MdDaudIbrahim/Ultra-Visual-Data-Analysis/main.yml?style=for-the-badge)
![License](https://img.shields.io/github/license/MdDaudIbrahim/Ultra-Visual-Data-Analysis?style=for-the-badge&color=blue)
![GitHub issues](https://img.shields.io/github/issues/MdDaudIbrahim/Ultra-Visual-Data-Analysis?style=for-the-badge&color=brightgreen)
![GitHub forks](https://img.shields.io/github/forks/MdDaudIbrahim/Ultra-Visual-Data-Analysis?style=for-the-badge&color=blueviolet)
![GitHub stars](https://img.shields.io/github/stars/MdDaudIbrahim/Ultra-Visual-Data-Analysis?style=for-the-badge&color=orange)

---

LIVE LINK-->  https://ultra-visual-data-analysis.streamlit.app/

> Transform your raw Excel or CSV data into stunning, interactive dashboards and actionable insights—no code required. This powerful Streamlit application provides a comprehensive suite for data profiling, quality assessment, and advanced visualization.

This tool is designed for data analysts, business professionals, students, and anyone who wants to quickly understand and visualize their datasets without writing complex scripts. Just upload your file and let the AI-powered analytics guide you.

## ✨ Key Features

*   **📈 Interactive Dashboard**: Get an instant overview of your data with dynamic metric cards and key performance indicators.
*   **🔍 Data Quality Assessment**: Automatically detect and analyze missing values, duplicate rows, and data completeness.
*   **🧠 AI-Powered Insights**: Receive intelligent observations, including high cardinality alerts, data skewness warnings, and outlier reports.
*   **🎨 Extensive Visualization Gallery**: Generate over 20 types of interactive charts, from basic distributions to advanced 3D plots, heatmaps, and treemaps.
*   **🔬 Advanced Statistical Analysis**: Dive deep with correlation matrices, descriptive statistics, outlier detection (IQR method), and trend analysis.
*   **🗂️ Smart Data Explorer**: Browse, search, and filter your dataset in real-time with an intuitive table view.
*   **💾 Easy Export**: Download cleaned data, analysis reports, and high-resolution charts with a single click.

---

## 📸 Showcase

*A picture is worth a thousand words. Here's a glimpse of what Ultra-Visual Data Analysis can do.*

**Main Dashboard & Data Overview**
<img width="960" height="446" alt="image" src="https://github.com/user-attachments/assets/28e3c6cd-cfe6-4967-bfcc-693863d01fe2" />
<img width="1916" height="813" alt="image" src="https://github.com/user-attachments/assets/35ba9ac7-0392-428a-a171-bd43ba749e05" />
<img width="1882" height="833" alt="image" src="https://github.com/user-attachments/assets/4579f3cb-6440-46cb-8899-a04f779f9b2b" />
<img width="1899" height="819" alt="image" src="https://github.com/user-attachments/assets/a5700948-7e76-4993-8879-055d04850844" />
<img width="1920" height="814" alt="image" src="https://github.com/user-attachments/assets/6dd6716b-6131-4ad1-bdba-e6e7b0a887d6" />


<img width="1802" height="756" alt="image" src="https://github.com/user-attachments/assets/30a5e189-8430-49e6-a204-3d5503ae64e5" />
<img width="1845" height="791" alt="image" src="https://github.com/user-attachments/assets/5510a4d0-0cd7-4a93-b3e8-f3dabaf513a8" />
<img width="950" height="452" alt="image" src="https://github.com/user-attachments/assets/964a740c-e921-4720-9bf5-6aee6b764708" />
<img width="1907" height="808" alt="image" src="https://github.com/user-attachments/assets/50be34d2-346e-45a9-857a-c0e279eabfcc" />



## 🛠️ Tech Stack & Tools

This project is built with a modern, powerful stack for data science and web development:

*   **Web Framework**:
    *   `Streamlit`
*   **Core Data Processing**:
    *   `Pandas`
    *   `NumPy`
*   **Visualization**:
    *   `Plotly`
    *   `Seaborn`
    *   `Matplotlib`
*   **Statistical Analysis**:
    *   `SciPy`
    *   `Scikit-learn`
*   **File I/O**:
    *   `Openpyxl`
    *   `XlsxWriter`

---

## 🚀 Getting Started

You can get the application running on your local machine in just a few minutes.

### Prerequisites

*   Python 3.8+
*   `pip` package manager

### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/MdDaudIbrahim/Ultra-Visual-Data-Analysis.git
    cd Ultra-Visual-Data-Analysis
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Project

The repository contains a few different versions of the application. The `ultra_visual_analyzer.py` is the most feature-complete version.

1.  **Run the main application:**
    ```bash
    streamlit run ultra_visual_analyzer.py
    ```

2.  Your web browser should automatically open to `http://localhost:8501`.

3.  **For Windows Users (One-Click Launch):**
    You can also use the provided batch files for convenience:
    *   Double-click `launcher.bat` and select option **3** for the "Ultra Visual Analyzer".
    *   Or, directly double-click `start_ultra_visual.bat`.

---

## 📖 Usage

1.  Launch the application using one of the methods described above.
2.  Use the sidebar to **upload your data file**.
    *   Supported formats: Excel (`.xlsx`, `.xls`) and CSV (`.csv`).
    *   Maximum recommended size: 200MB.
3.  Once uploaded, the dashboard will populate automatically.
4.  Use the sidebar checkboxes to toggle different **analysis modules** (e.g., Data Quality, AI Insights, Statistical Analysis).
5.  Explore the various tabs and interactive charts to gain insights from your data.

---

## 📁 Project Structure

Here's a brief overview of the key files in this repository:

```
Ultra-Visual-Data-Analysis/
├── .streamlit/             # Configuration for Streamlit Cloud
├── requirements.txt        # List of Python dependencies
├── streamlit_app.py        # Entry point for Streamlit Cloud deployment
├── ultra_visual_analyzer.py# The main, most feature-rich application
├── advanced_excel_analyzer.py # An advanced version of the analyzer
├── excel_analyzer.py       # A more basic version of the analyzer
├── advanced_analyzer.py    # Core class with data analysis logic
├── *.bat                   # Windows batch files for easy launching
└── USER_GUIDE.md           # Detailed guide on using the features
```

---

## 🤝 Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

Please see `CONTRIBUTING.md` for more details on our code of conduct and the process for submitting pull requests.

---

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for more information.

---

## 🙏 Acknowledgements

*   The vibrant communities behind Streamlit, Plotly, and Pandas.
*   All contributors who help improve this project.

<br>

**Star this repository if you find it useful! ⭐**
