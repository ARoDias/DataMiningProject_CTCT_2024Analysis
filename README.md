# DataMiningProject_CTCT_2024Analysis

This repository contains Python code for analyzing CTCT data. The project includes notebooks for data preparation, feature selection, clustering, prediction modeling, and an in-depth data understanding phase based on the CRISP-DM methodology.

## Project Structure

- `data_preparation.ipynb`: Notebook for cleaning and preparing the dataset.
- `feature_selection_and_clustering.ipynb`: Feature selection and clustering methods. Answers **RQ4** and selects features for **RQ5**.
- `CTCT_Data_Mining_Prediction_and_Classification.ipynb`: Main notebook for prediction and classification. Focuses on **RQ5**.
- `Data_Analysis_Report.ipynb`: Notebook for generating the final report and analysis summary.
- `data_understanding.ipynb`: CRISP-DM data understanding phase, including descriptive statistics, correlation analysis, statistical testing, and visualizations. Primarily answers **RQ1**, **RQ2**, and **RQ3**.

## Research Questions

This project focuses on answering the following research questions:

1. **RQ1**: How does gender impact course selection among students across different academic years?
2. **RQ2**: What are the interrelationships and correlation strengths between Activity Points (AP), Participation Points (PP), and Self and Peer Assessment Points (SPA) across various academic years? And how do these assessment points correlate to the final grade in the discipline?
3. **RQ3**: How does student performance in the CTCT discipline vary across editions, students' degrees (courses), gender (sex), weeks, and themes, as measured by average points?
4. **RQ4**: What distinct clusters are evident in student performances based on different types of assessment points?
5. **RQ5**: Can student performance in one theme of the CTCT course predict outcomes in subsequent themes?

## Notebooks Overview

### Data Preparation Notebook

- **File**: `data_preparation.ipynb`
- **Description**: Prepares and cleans the raw dataset for analysis. Handles missing data, normalizes feature values, and organizes the dataset to be used in subsequent notebooks.

### Data Understanding Notebook

- **File**: `data_understanding.ipynb`
- **Description**: Provides an in-depth understanding of the dataset, including:
  - Descriptive statistics
  - Correlation analysis
  - Statistical testing (Chi-squared tests, Mann-Whitney U tests, Shapiro-Wilk tests)
  - Various visualizations (histograms, pie charts, heatmaps, and bar charts)
- **Research Questions Answered**: Primarily addresses **RQ1**, **RQ2**, and **RQ3**.

### Feature Selection and Clustering Notebook

- **File**: `feature_selection_and_clustering.ipynb`
- **Description**: 
  - Implements feature selection techniques to determine which features are most relevant for predicting outcomes.
  - Applies clustering algorithms to identify distinct student performance groups.
- **Research Questions Answered**: 
  - Answers **RQ4**: "What distinct clusters are evident in student performances based on different types of assessment points?"
  - Performs feature selection for **RQ5**: "Can student performance in one theme of the CTCT course predict outcomes in subsequent themes?"

### Prediction and Classification Notebook

- **File**: `CTCT_Data_Mining_Prediction_and_Classification.ipynb`
- **Description**: 
  - Implements machine learning models for predicting student outcomes in later stages of the CTCT course based on their performance in earlier stages.
- **Research Questions Answered**: 
  - Focuses on **RQ5**: "Can student performance in one theme of the CTCT course predict outcomes in subsequent themes?"

### Data Analysis Report Notebook

- **File**: `Data_Analysis_Report.ipynb`
- **Description**: Generates a summary report of the analysis, aggregating the results from all previous stages. This notebook includes visualizations, performance metrics, and conclusions drawn from the models.

## Google Colab Usage

All notebooks in this repository are designed to be run on [Google Colab](https://colab.research.google.com/), where all necessary dependencies can be installed directly within the notebook.

To use these notebooks on Colab:

1. Open the desired notebook by clicking on the Colab badge (if provided) or uploading the notebook file to Google Colab.
2. The necessary Python packages will be installed using the code blocks within each notebook. Example:
   ```python
   !pip install numpy pandas scikit-learn matplotlib
   ```
Note: You do not need to manually install dependencies unless you are running the notebooks in another environment (e.g., Anaconda).
   
## Installation (Local Environment - Optional)
If you prefer to run the notebooks locally (e.g., in an Anaconda environment), you can clone the repository and install the dependencies manually. To do this:

## Clone this repository:
git clone https://github.com/your-username/DataMiningProject_CTCT_2024Analysis.git

## Install the required dependencies (if using a local environment):
pip install -r requirements.txt
Run the notebooks in Jupyter or any Python environment.

## Data
Note: The actual CTCT dataset is not included in this repository for privacy reasons. Please ensure you have the necessary data files before running the notebooks. Data should be in the expected format (e.g., .csv, .pickle, .feather), which will be loaded as described in each notebook.

## Outputs
All notebooks are configured to save their outputs (e.g., results, predictions, and analysis) into local files such as .csv, .xlsx, or .docx. This ensures that the results of the analysis are properly documented and can be easily accessed for further use.

## Research Questions
The notebooks aim to answer several research questions, including but not limited to:

RQ1: How does gender impact course selection among students across different academic years?
RQ2: What are the interrelationships and correlation strengths between Activity Points (AP), Participation Points (PP), and Self and Peer Assessment Points (SPA) across various academic years? And how do these assessment points correlate to the final grade in the discipline?
RQ3: How does student performance in the CTCT discipline vary across editions, students' degrees (courses), gender (sex), weeks, and themes, as measured by average points?

## Visualization of Results
Visualization plays a key role in understanding the data. The notebooks generate various visualizations to facilitate analysis:

Histograms: To visualize the distribution of features and identify patterns or outliers.
Pie Charts: To show the composition of categorical variables.
Correlation Heatmaps: To visualize relationships between numerical variables.
Bar Charts: To compare groups or track changes over time.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
