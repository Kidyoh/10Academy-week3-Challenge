# AI Mastery Challenge Week 3

Welcome to the AI Mastery Challenge for Week 3! This repository contains the work focused on data analytics in the insurance domain, specifically aimed at optimizing marketing strategies and understanding risk factors.

## Project Overview

The goal of this project is to analyze historical insurance claim data to help AlphaCare Insurance Solutions (ACIS) optimize their marketing strategy and identify low-risk targets for premium reductions. The project involves various tasks, including exploratory data analysis (EDA), A/B hypothesis testing, statistical modeling, and machine learning.

## Project Structure

- **data/**: Contains raw, processed, and external data.
  - `MachineLearningRating_v3.txt.dvc`: DVC file for tracking the Machine Learning Rating dataset.
  
- **notebooks/**: Jupyter notebooks for exploratory data analysis and modeling.
  - `__init__.py`: Initializes the notebooks package.

- **scripts/**: Contains Python scripts for various tasks.
  - `eda_task_1.py`: Script for performing exploratory data analysis for Task 1.
  - `ab_testing.py`: Script for conducting A/B hypothesis testing.
  - `src/__init__.py`: Initializes the source package.

- **reports/**: Documentation and reports.
  - `interim_report.md`: Report covering the interim findings.
  - `final_report.md`: Final report summarizing all work done.

- **requirements/**: Dependency management files.
  - `requirements.txt`: Lists Python package dependencies.
  - `environment.yml`: Conda environment file for setting up the project.

- **tests/**: Unit tests for the project.
  - `test_data.py`: Tests for data processing functions.
  - `test_models.py`: Tests for machine learning models.
  - `test_utils.py`: Tests for utility functions.

- **.github/**: GitHub Actions workflows for CI/CD.
  - `workflows/unittests.yml`: Workflow for running unit tests and linting.

- **.dvc/**: Configuration for Data Version Control.
  - `.gitignore`: Specifies files and directories to ignore in DVC.

- **main.py**: Main entry point for running the project.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Conda (optional, if using the environment.yml)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/AI_Mastery_Challenge_Week3.git
   cd AI_Mastery_Challenge_Week3
   ```

2. Install dependencies:
   - Using pip:
     ```bash
     pip install -r requirements/requirements.txt
     ```
   - Or using conda:
     ```bash
     conda env create -f requirements/environment.yml
     ```

### Running the Project

- To run the main script:
  ```bash
  python main.py
  ```

- To execute the exploratory data analysis:
  ```bash
  python scripts/eda_task_1.py
  ```

- To conduct A/B testing:
  ```bash
  python scripts/ab_testing.py
  ```

### Running Tests

To run the unit tests, you can use the following command:
```

### CI/CD

This project uses GitHub Actions for continuous integration. The unit tests and linting are automatically run on every push and pull request to the `main` branch.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to 10 Academy for providing the framework for this challenge.
- Special thanks to the contributors and mentors who provided guidance throughout the project.

---

Feel free to reach out if you have any questions or need further assistance!