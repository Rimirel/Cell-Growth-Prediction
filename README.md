# Cell Growth Rate Prediction Using Machine Learning

Predict specific cell growth rates from temperature, pH, glucose, and lactate concentrations using machine learning regression models.

# Cell Growth Rate Prediction

This project develops and evaluates machine learning regression models to predict the specific growth rate of cells from four process variables: temperature, pH, glucose concentration, and lactate concentration. The workflow combines data preparation, model training, independent testing, and five-fold cross-validation. Polynomial regression, K-nearest neighbors (KNN), and Random Forest are compared using quantitative performance metrics to identify promising approaches for modeling cell growth behavior.

## Highlights

- **Build** regression models for cell growth prediction.
- **Compare** polynomial, KNN, and Random Forest algorithms.
- **Evaluate** predictions using R², MAE, and RMSE.
- **Validate** model performance through five-fold cross-validation.
- **Analyze** the effects of four process variables.

## Environment

The project uses Python and the following libraries. 
- Python 3.14.7
- NumPy 2.5.2
- pandas 3.0.5
- scikit-learn 1.9.0


## Code Design

### 1. Project Organization and Configuration

The project separates configuration, model creation, model evaluation, and execution into distinct files. This organization improves readability, makes the implementation easier to maintain, and allows individual components to be modified without rewriting the entire workflow.

The `datasets/` directory contains the input file, `dataset_cell_growth_4factor.csv`. The `src/` directory contains the Python modules used by the main script. The `main.py` file coordinates the complete workflow, from loading the dataset to generating performance reports.

The `src/constants.py` module defines the dataset path, the four input features, the target variable, the test-set proportion, and the random seed. Centralizing these values avoids repeating configuration settings across the code and makes future adjustments easier. The target variable is `Specific Growth Rate [d^-1]`, while the input features are `Temperature [C]`, `pH`, `Glucose [mM]`, and `Lactate [mM]`.

### 2. Data Loading and Validation

The workflow begins by locating the project directory and loading the CSV file with pandas. The code uses the project directory to construct the dataset path, making file access less dependent on the working directory configured in the development environment.

Before model training, the script verifies that all required columns are present. If any required column is missing, the program raises a descriptive error rather than continuing with an incorrectly structured dataset. The selected columns are then extracted, and rows containing missing values are removed.

The feature matrix, \(X\), contains the four process variables, while the target vector, \(y\), contains the measured specific growth rate. This separation follows the standard structure of a supervised regression problem, in which the model learns a relationship between explanatory variables and a continuous response.

### 3. Training and Testing Strategy

The prepared dataset is divided into training and testing subsets using `train_test_split` from scikit-learn. The test-set proportion is configured as 20%, and the random seed is set to 42.

The training subset is used to fit the models, while the independent test subset is reserved for evaluating their predictions. Using a fixed random seed makes the split reproducible, allowing the same experiment to be repeated under consistent conditions.

Keeping test data separate from model fitting provides a more independent assessment of performance on unseen observations. This is important because a model can fit its training data well without necessarily predicting new observations accurately.

### 4. Regression Model Construction

The `create_models()` function in `src/functions.py` creates three regression pipelines.

Polynomial regression uses `PolynomialFeatures` with degree two, followed by `StandardScaler` and `LinearRegression`. The polynomial transformation allows the model to represent quadratic effects and interactions between input variables. Scaling is included to standardize the transformed features.

K-nearest neighbors regression uses standardized input features and `KNeighborsRegressor` with five neighbors and distance-based weighting. The model estimates a prediction from nearby observations, giving greater influence to closer neighbors.

Random Forest uses 200 decision trees and a minimum of two samples per leaf. Its random seed is also fixed at 42. By combining multiple decision trees, Random Forest can capture nonlinear relationships and interactions without requiring an explicit polynomial transformation.

The three approaches were selected to compare different ways of representing relationships between process variables and cell growth. Polynomial regression provides a relatively interpretable baseline, KNN uses local similarity, and Random Forest learns nonlinear patterns through an ensemble of trees.

### 5. Model Evaluation on the Test Set

The `ModelEvaluator` class in `src/classes.py` handles training and evaluation. For each model, the class fits the model using the training data, generates predictions for the test data, and calculates three performance metrics.

The coefficient of determination, R², measures how much of the variation in the target is explained by the predictions relative to a mean-based baseline. Mean absolute error (MAE) measures the average absolute difference between predicted and observed growth rates. Root mean squared error (RMSE) measures the square root of the average squared prediction error, giving relatively greater weight to larger errors.

The results are collected into a pandas DataFrame, sorted by RMSE, displayed in the terminal, and saved to `model_comparison.csv`. This provides a consistent basis for comparing the three algorithms.

### 6. Five-Fold Cross-Validation

The `compare_models_cv()` function evaluates the models using five-fold cross-validation. The dataset is divided into five folds, and each model is evaluated repeatedly, using four folds for training and the remaining fold for validation in each iteration.

The folds are shuffled before splitting, and the random seed is fixed at 42. For each model, the mean R², mean MAE, and mean RMSE are calculated across the five validation folds. Because scikit-learn returns negative values for the selected error scorers, the MAE and RMSE values are converted back to positive values before reporting.

The averaged metrics are saved to `cross_validation_results.csv`. This additional evaluation helps assess whether a model's performance depends strongly on a single train-test split. It complements the independent test-set evaluation rather than replacing it.

### 7. Output and Reproducibility

The `main.py` script integrates the complete workflow and produces two CSV reports. `model_comparison.csv` contains the independent test-set results, while `cross_validation_results.csv` contains the mean metrics from five-fold cross-validation.

The separation of inputs, configuration, model creation, evaluation, and execution makes the project easier to reproduce and extend. For example, additional regression algorithms, alternative hyperparameters, or new evaluation metrics can be introduced without substantially changing the overall structure.

## Analysis

### 1. Modeling Objective and Dataset

This project predicts specific cell growth rate from temperature, pH, glucose concentration, and lactate concentration. The dataset contains 318 simulated observations. Since the target, `Specific Growth Rate [d^-1]`, is continuous, the task is a supervised regression problem.

The dataset is simulated, so the results show how the selected models perform on these data; they do not guarantee similar accuracy with experimental measurements or in an industrial bioreactor.

### 2. Model Selection and Hyperparameters

Three regression algorithms were compared: Polynomial Regression, K-nearest neighbors (KNN), and Random Forest.

#### Polynomial Regression

Polynomial Regression uses degree-two polynomial features followed by standardization and linear regression. The quadratic terms and pairwise interactions allow it to represent some nonlinear relationships while remaining relatively interpretable. However, a fixed quadratic structure may not capture more complex patterns.

#### K-Nearest Neighbors Regression

KNN uses five neighbors with distance-based weighting. Standardization is important because KNN relies on distances and the input variables have different scales. KNN can capture local nonlinear patterns, but its performance depends on the number of neighbors and the distribution of observations.

#### Random Forest Regression

Random Forest combines 200 decision trees and uses a minimum leaf size of two observations. It can learn nonlinear relationships and interactions without explicitly specifying their form. However, it is less directly interpretable, and the chosen hyperparameters were not exhaustively optimized.

These settings provide a comparison of three different modeling approaches, rather than the best possible configuration of each algorithm.

### 3. Evaluation Metrics

The models were assessed using R², mean absolute error (MAE), and root mean squared error (RMSE). Higher R² values generally indicate better performance, while lower MAE and RMSE indicate smaller prediction errors. MAE represents the average absolute error; RMSE gives relatively greater weight to larger errors. Both error metrics are expressed in the target's units, d⁻¹.

### 4. Independent Test-Set Results

The dataset was divided into training and testing subsets using an 80:20 split.

| Model | R² | MAE | RMSE |
|---|---:|---:|---:|
| Polynomial Regression | 0.221390 | 0.117278 | 0.145497 |
| Random Forest | 0.189904 | 0.115946 | 0.148410 |
| KNN Regression | 0.176989 | 0.122969 | 0.149588 |

Polynomial Regression achieved the highest R² and lowest RMSE on this test split. Random Forest achieved the lowest MAE, approximately 0.116 d⁻¹. KNN had the lowest R² and the highest error values among the three models.

The R² of Polynomial Regression (approximately 0.221) indicates that it explains about 22% of the variation in the test-set target relative to a mean-based baseline; it does not mean that 22% of predictions are correct. Overall, the R² values are modest, so substantial variation remains unexplained. These rankings also reflect one particular test split and may change with a different split.

### 5. Five-Fold Cross-Validation Results

Five-fold cross-validation was used to compare performance across multiple train-validation splits.

| Model | Mean R² | Mean MAE | Mean RMSE |
|---|---:|---:|---:|
| Random Forest | 0.325573 | 0.123874 | 0.152569 |
| Polynomial Regression | 0.319398 | 0.124776 | 0.153748 |
| KNN Regression | 0.287788 | 0.127696 | 0.156686 |

Random Forest achieved the highest mean R², approximately 0.326. Polynomial Regression had slightly lower mean MAE and RMSE, while KNN had the weakest results across the three metrics. The differences between Random Forest and Polynomial Regression are small, and the preferred model depends partly on which metric is prioritized.

The rankings differ from the test-set comparison, showing why performance should not be judged from one split alone. Reporting the standard deviation across folds would provide additional information about stability. Since the hyperparameters were not extensively tuned, these results compare the selected configurations rather than each algorithm's maximum potential.

### 6. Comparison of Model Advantages and Limitations

Polynomial Regression is relatively simple and performed best on the test set in terms of R² and RMSE. It also had the lowest mean RMSE in cross-validation, but its quadratic form may not represent more complex growth behavior.

Random Forest can capture nonlinear relationships and interactions. It achieved the highest mean R² in cross-validation and the lowest test-set MAE, although its RMSE was slightly higher than Polynomial Regression's. Tuning parameters such as tree depth and minimum leaf size may improve its performance.

KNN offers a local, distance-based approach but performed least favorably with the selected configuration. Its performance may depend on the neighbor count, distance metric, feature scaling, and density of observations. Further tuning would be needed before drawing a firm conclusion about its suitability.

Overall, Polynomial Regression and Random Forest are the strongest candidates in this comparison, but their modest R² values indicate that neither model captures all the variation in growth rate.

### 7. Influence of Temperature, pH, Glucose, and Lactate

Temperature, pH, glucose, and lactate were selected because they can be relevant to cellular growth. Temperature and pH may have suitable operating ranges, with growth decreasing when conditions move away from them. Glucose supplies a carbon and energy source, but higher concentrations do not necessarily produce proportionally higher growth. Lactate accumulation may be associated with less favorable conditions in some systems, although its effect depends on the organism and process.

The feature relationship plots show how each variable changes alongside measured growth rate, but they do not isolate the effect of one variable from the others. The current workflow does not calculate feature importance or establish causal relationships. Therefore, it cannot yet confirm which variable is the strongest predictor or specify how changing one variable will affect growth. Feature importance and controlled prediction plots could support a more detailed analysis.

### 8. High-Growth, Low-Growth, and Unreliable Predictions

Conditions associated with high or low growth should be identified from the dataset and model predictions rather than assumed in advance. Ranking observations by measured growth rate and comparing their input conditions could reveal combinations associated with higher or lower growth in these simulated data. Such patterns would not automatically establish optimal conditions for a real biological process.

Predictions may be unreliable for input combinations outside the ranges represented in the training data or where observations are sparse. KNN depends on nearby examples, Random Forest predictions are limited by the learned tree splits, and Polynomial Regression may produce unrealistic extrapolations far outside the observed range. Given the modest R² scores, predictions should be treated as estimates rather than precise measurements.

### 9. Practical Applications and Future Improvements

This type of model could support preliminary bioprocess analysis, experimental planning, and screening of operating conditions for further investigation. However, the current results are not sufficient for direct process-control decisions because the dataset is simulated, predictive performance is modest, and no independent experimental validation has been performed.

Future improvements could include tuning hyperparameters through cross-validation on the training data while reserving the test set for final evaluation, reporting cross-validation variability, and examining residuals for systematic errors. Feature-importance analysis and prediction plots could help investigate the roles of the four input variables. Most importantly, validation with independent experimental data would be required before applying the models to a real cultivation system.

### 10. Conclusion

Polynomial Regression achieved the best test-set R² and RMSE, while Random Forest achieved the highest mean R² in cross-validation and the lowest test-set MAE. Polynomial Regression had slightly lower mean MAE and RMSE during cross-validation. KNN performed less favorably with the selected settings.

The results show that model ranking depends on the evaluation metric and data split. Polynomial Regression and Random Forest are reasonable candidates for further investigation, but the modest R² values indicate that substantial variation remains unexplained. Hyperparameter tuning, further error analysis, and validation on independent experimental data are recommended before practical use.