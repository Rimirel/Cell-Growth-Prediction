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

The project uses Python and the following libraries. Replace the version placeholders with the versions installed in the environment used to run the project.

- Python
- NumPy 
- pandas 
- scikit-learn


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

Predicting specific growth rate is useful for understanding how environmental and nutrient conditions relate to cellular growth. In a bioprocess, temperature and pH influence cellular activity, while glucose availability and lactate accumulation can affect metabolism and growth. A predictive model can help identify patterns in these variables and support the analysis of operating conditions.

The dataset used in this project contains 318 observations and five columns. Four columns describe the input conditions: temperature in degrees Celsius, pH, glucose concentration in millimolar, and lactate concentration in millimolar. The fifth column is the specific growth rate in inverse days.

The prediction task is a supervised regression problem because the target is a continuous numerical variable. The objective is to estimate the growth rate associated with a combination of process conditions rather than assign observations to predefined categories.

The dataset is simulated. Consequently, the results demonstrate the behavior of the selected models on the available data, but they do not establish that the same performance will be achieved with experimental measurements or in an industrial bioreactor. The relationships learned by the models depend on the patterns, ranges, and assumptions represented in the dataset.

### 2. Model Selection and Hyperparameters

Three algorithms were selected to compare different modeling strategies: polynomial regression, K-nearest neighbors regression, and Random Forest regression.

#### Polynomial Regression

Polynomial regression was implemented using degree-two polynomial features followed by linear regression. The transformation introduces squared terms and pairwise interactions between the original features. This allows the model to represent some nonlinear behavior while retaining a relatively simple mathematical structure.

The degree was fixed at two to balance flexibility and simplicity. A higher degree could represent more complex patterns but could also introduce unnecessary complexity and increase the risk of overfitting. StandardScaler was included after polynomial feature generation so that the transformed features are placed on comparable scales. This is particularly relevant when feature magnitudes differ.

The main advantage of polynomial regression is its ability to represent smooth quadratic relationships and interactions in a relatively interpretable form. However, its structure may be insufficient when growth behavior depends on more complex nonlinear relationships, thresholds, or interactions that cannot be adequately represented by a quadratic function.

#### K-Nearest Neighbors Regression

KNN predicts the growth rate of an observation using the target values of nearby training observations. The implementation uses five neighbors and distance-based weighting. Nearby observations therefore contribute more strongly to the prediction than observations farther away.

StandardScaler is used because KNN relies on distances between observations. Without scaling, variables with larger numerical ranges could dominate the distance calculation, even when they are not biologically more important.

Using five neighbors provides a compromise between relying on very few observations and averaging over a larger neighborhood. However, this value was specified rather than established through a comprehensive hyperparameter search. Different neighbor counts, distance metrics, or weighting schemes could change the results.

KNN can represent local nonlinear relationships without requiring a predefined equation. Its limitations include sensitivity to feature scaling, the distribution of training observations, and the selected neighborhood size. Predictions may also become unreliable in regions where training observations are sparse.

#### Random Forest Regression

Random Forest combines predictions from multiple decision trees. Each tree learns relationships between the input variables and growth rate, and the ensemble combines their predictions. This approach can represent nonlinear patterns and interactions without explicitly constructing polynomial terms.

The implementation uses 200 trees and a minimum leaf size of two observations. The ensemble size provides multiple trees for aggregation, while the minimum leaf size helps limit overly specific splits. A fixed random seed supports reproducibility.

Random Forest is attractive for data containing nonlinear effects or interactions that are difficult to describe with a simple equation. However, its predictions are less directly interpretable than those of a low-degree polynomial model, and the selected settings do not guarantee optimal performance. Additional tuning would be needed to establish whether other values improve generalization.

### 3. Evaluation Metrics

Model performance is evaluated using R², MAE, and RMSE. These metrics provide complementary information, so no single metric should be interpreted in isolation.

R² measures predictive performance relative to a baseline that predicts the mean target value. A value closer to one generally indicates better performance, while a value near zero indicates performance similar to that baseline under the evaluation conditions. Negative values are also possible when predictions perform worse than the mean-based baseline.

MAE represents the average magnitude of absolute prediction errors. RMSE also measures prediction error but penalizes larger deviations more strongly because errors are squared before averaging. Both metrics use the same units as the target, here days⁻¹, which makes their magnitudes easier to interpret.

For R², higher values are preferred. For MAE and RMSE, lower values are preferred. Considering all three metrics provides a more balanced comparison, particularly when two models rank differently depending on the error measure.

### 4. Independent Test-Set Results

The dataset was split into training and testing subsets using an 80:20 ratio and a fixed random seed. The resulting test-set metrics are shown below.

| Model | R² | MAE | RMSE |
|---|---:|---:|---:|
| Polynomial Regression | 0.221390 | 0.117278 | 0.145497 |
| Random Forest | 0.189904 | 0.115946 | 0.148410 |
| KNN Regression | 0.176989 | 0.122969 | 0.149588 |

Polynomial regression achieved the highest R² and the lowest RMSE on this particular test split. Its R² of approximately 0.221 indicates that it explains about 22% of the variation in the test-set target values relative to the mean-based baseline. This does not mean that 22% of individual predictions are correct.

Random Forest achieved the lowest MAE, at approximately 0.116 d⁻¹. Its RMSE was slightly higher than that of polynomial regression, suggesting that its overall error profile differs depending on how errors are weighted. KNN had the lowest R² and the highest MAE and RMSE among the three models in this test evaluation.

The differences are relatively small in absolute error, and the R² values are modest. Therefore, the test results do not indicate that any model provides highly accurate predictions. They suggest that the models capture some information about growth rate, but a substantial amount of variation remains unexplained.

The test-set ranking should also be interpreted cautiously because it is based on a single data split. A different split could change the relative performance of the models. This motivates the additional cross-validation analysis.

### 5. Five-Fold Cross-Validation Results

Five-fold cross-validation was used to evaluate the models over multiple train-validation splits. The mean results are shown below.

| Model | Mean R² | Mean MAE | Mean RMSE |
|---|---:|---:|---:|
| Random Forest | 0.325573 | 0.123874 | 0.152569 |
| Polynomial Regression | 0.319398 | 0.124776 | 0.153748 |
| KNN Regression | 0.287788 | 0.127696 | 0.156686 |

Random Forest achieved the highest mean R², approximately 0.326. Polynomial regression ranked second on mean R² but obtained slightly lower mean MAE and RMSE than Random Forest. KNN had the weakest mean performance across all three metrics.

The difference between Random Forest and polynomial regression is small. Random Forest explains more variation on average, while polynomial regression has a slight advantage in average prediction error. The preferred model therefore depends partly on whether explained variation or error magnitude is prioritized.

The rankings differ from those obtained on the independent test set. Polynomial regression performed best in terms of R² and RMSE on that split, whereas Random Forest achieved the highest mean R² during cross-validation. This difference illustrates why evaluating models on multiple splits is valuable. It also means that the apparent winner should not be selected solely from the independent test results.

Cross-validation provides a broader assessment of performance, but the reported mean metrics do not show the variability across folds. Reporting the standard deviation of each metric would provide additional information about stability. Moreover, because the hyperparameters were not exhaustively tuned, the results represent a comparison of the specified configurations rather than the best achievable performance of each algorithm.

### 6. Comparison of Model Advantages and Limitations

Polynomial regression offers a relatively simple structure and can represent quadratic relationships and pairwise interactions. It performed best on the independent test split according to R² and RMSE and had the lowest mean RMSE in cross-validation. These results make it a reasonable baseline and a competitive candidate when prediction error and simplicity are important. Its main limitation is that its fixed quadratic form may not capture more complicated growth behavior.

Random Forest achieved the highest mean R² in cross-validation and the lowest MAE on the independent test set. It can learn nonlinear relationships and interactions without specifying their mathematical form in advance. However, it showed slightly higher RMSE than polynomial regression in both reported comparisons. Its performance could potentially be improved by tuning parameters such as tree depth, minimum samples per leaf, and the number of features considered at each split.

KNN provides a local, distance-based alternative. It can capture nonlinear patterns when similar observations have similar growth rates. However, it had the weakest performance in the current comparison. This may reflect the chosen number of neighbors, the structure of the dataset, the distance metric, or the difficulty of estimating growth rates in sparse regions. Further tuning would be required before concluding that KNN is inherently unsuitable for this problem.

Overall, polynomial regression and Random Forest are the most promising candidates among the configurations tested. The small differences between them and the modest R² values suggest that further validation and model improvement are needed before making a definitive selection.

### 7. Influence of Temperature, pH, Glucose, and Lactate

Temperature, pH, glucose, and lactate are the four input variables used to predict specific growth rate. Each can plausibly relate to cellular growth, but the direction and magnitude of these relationships depend on the organism, medium, and operating conditions represented by the data.

Temperature affects reaction rates, enzyme activity, membrane function, and other cellular processes. Growth can increase as temperature approaches a suitable range, but performance may decline outside that range. Consequently, the relationship between temperature and growth rate may be nonlinear rather than consistently increasing.

pH influences enzyme activity, nutrient transport, and cellular homeostasis. Cells generally have a range of pH conditions that supports growth, and deviations from this range can affect their performance. A model may capture an optimum-like relationship if the dataset contains sufficient observations across the relevant pH range.

Glucose provides a carbon and energy source for many cell cultures. Increasing glucose availability can support growth when glucose is limiting, but the relationship may not remain positive at all concentrations. Depending on the system, high concentrations or associated metabolic effects can alter growth behavior.

Lactate can be associated with cellular metabolism and may accumulate during cultivation. In some systems, elevated lactate is associated with less favorable growth conditions, although its effect depends on the organism and process context. It should not automatically be interpreted as having a negative effect in every observation.

These biological considerations help motivate the choice of input variables, but they do not establish the actual effects learned from this dataset. The current workflow evaluates predictive performance; it does not yet calculate feature importance, partial dependence, or the direction of each variable's influence. Therefore, it would be premature to claim that a specific variable is the strongest predictor or that increasing or decreasing it always improves growth.

A more detailed analysis could use permutation importance for Random Forest, model coefficients for polynomial regression, and controlled prediction plots across the observed ranges of each feature. Interactions should also be investigated because the effect of one variable may depend on the levels of the others. Such analyses would help distinguish patterns supported by the data from general biological expectations.

### 8. High-Growth, Low-Growth, and Unreliable Predictions

Conditions associated with high or low predicted growth should be identified using the dataset and the fitted models rather than assumed in advance. A practical approach is to rank observations by their measured growth rate, compare their input conditions, and inspect whether similar patterns appear in model predictions. This could reveal combinations of temperature, pH, glucose, and lactate associated with higher or lower growth in the available data.

However, such patterns should be interpreted within the range of the observations. A combination associated with high growth in the simulated dataset is not automatically an optimal condition for a real biological process. Confirmation would require relevant experimental evidence and knowledge of the underlying system.

Predictions may be unreliable when the input conditions fall outside the ranges represented in the training data. This is especially important for KNN, which depends on nearby observations, and for tree-based models, which do not extrapolate smoothly beyond learned split regions. Polynomial regression can generate extrapolated values, but these may become unrealistic far outside the observed range.

Predictions can also be uncertain when the dataset contains few examples for a particular combination of conditions, when relationships are highly variable, or when the model's errors are large. Since the current R² scores are modest, the predictions should be treated as estimates rather than precise measurements. A future version could report prediction errors, identify out-of-range inputs, and compare the stability of predictions across validation folds.

### 9. Practical Applications and Future Improvements

A model that predicts specific growth rate from process conditions could eventually support preliminary process analysis, experimental planning, and screening of candidate operating conditions. For example, it could help identify combinations of temperature, pH, glucose, and lactate that merit further experimental investigation. It could also serve as a baseline for comparing more advanced predictive approaches.

Nevertheless, the present results are not sufficient to justify direct use for process control or operational decisions. The dataset is simulated, the models explain only a limited proportion of the target variation, and no external experimental validation has been performed. Any real-world use would require validation against data from the relevant organism and cultivation system.

Several improvements could strengthen the analysis. First, hyperparameter tuning could be conducted using cross-validation on the training data, while keeping the test set untouched until final evaluation. Second, the cross-validation results could include standard deviations to show the variability of performance across folds. Third, residual plots and predicted-versus-observed plots could reveal systematic errors, extreme deviations, or regions where predictions are less accurate.

Feature importance and partial dependence analysis could help investigate how the four input variables influence model predictions. Additional algorithms could also be tested, provided that their performance is assessed using the same evaluation strategy. Finally, validation on independent experimental data would be necessary to determine whether the observed relationships generalize beyond the simulated dataset.

### 10. Conclusion

This project compared polynomial regression, KNN regression, and Random Forest regression for predicting specific cell growth rate from four process variables. Polynomial regression achieved the best R² and RMSE on the independent test split, while Random Forest achieved the highest mean R² in five-fold cross-validation and the lowest MAE on the test set. Polynomial regression had slightly lower mean MAE and RMSE during cross-validation.

The results show that the choice of model depends on the evaluation metric and data split. Both polynomial regression and Random Forest are reasonable candidates for further investigation, while KNN performed less favorably with the selected configuration. However, all three models achieved modest R² values, indicating that substantial variation in growth rate remains unexplained.

Further hyperparameter optimization, residual analysis, feature-effect investigation, and validation on independent experimental data are recommended before using the models for practical bioprocess decisions.