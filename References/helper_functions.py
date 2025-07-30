

import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import textwrap
import contractions
import re

from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split, cross_val_score
from statsmodels.formula.api import ols
from matplotlib.lines import Line2D
from imblearn.over_sampling import SMOTE
from statsmodels.stats.diagnostic import het_breuschpagan
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from tabulate import tabulate
from nltk.corpus import stopwords
from textblob import TextBlob
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from afinn import Afinn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline


def get_regression_results_with_plot(dep_var, indep_vars, data, outliers_removed=False, weighted=False,
                                     robust_var = 'nonrobust'):
    # Create a copy of the original data to avoid modifying it directly
    df = data.copy()

    # Ensure indep_vars is a list even if a single string is provided
    if isinstance(indep_vars, str):
        indep_vars = [indep_vars]

    if outliers_removed:
        # Function to remove outliers using the IQR method
        def remove_outliers(df, column):
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

        # Remove outliers for each variable in the model
        for column in [dep_var] + indep_vars:
            df = remove_outliers(df, column)

    # Determine the grid size based on the number of independent variables
    n_plots = len(indep_vars)
    if n_plots == 1:
        n_cols, n_rows = 1, 1
    else:
        n_cols = 2
        n_rows = (n_plots + 1) // 2  # Calculate the number of rows needed

    # Create subplots with the determined grid size
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows))

    # Flatten axes array for easy iteration
    axes = axes.flatten() if n_plots > 1 else [axes]

    # Dictionary to store models
    models = {}

    for i, indep_var in enumerate(indep_vars):
        # Create the formula string for the current independent variable
        formula = f'{dep_var} ~ {indep_var}'

        if weighted:
            # Fit an initial OLS model to get residuals
            initial_model = ols(formula=formula, data=df).fit(cov_type = robust_var)
            residuals = initial_model.resid
            # Calculate weights as the inverse of the squared residuals
            weights = 1 / (np.abs(residuals) + np.finfo(float).eps)
            # Fit the weighted linear regression model
            model = sm.WLS.from_formula(formula, data=df, weights=weights).fit()
        else:
            # Fit the linear regression model
            model = ols(formula=formula, data=df).fit(cov_type= robust_var)

        # Store the model in the dictionary
        models[indep_var] = model

        # Extract intercept, coefficient, and R-squared for the legend
        intercept = model.params['Intercept']
        coefficient = model.params[indep_var]
        r_squared = model.rsquared
        regression_line = f"y = {intercept:.2f} + {coefficient:.2f} * x"
        legend_label = f"{regression_line}\nR² = {r_squared:.2f}"

        # Scatter plot
        sns.scatterplot(data=df, x=indep_var, y=dep_var, ax=axes[i], label='Data')

        # Sort the values for the regression line
        sort_df = df.sort_values(by=indep_var)
        x_values = sort_df[indep_var]
        y_pred = model.predict(sort_df)

        # Plot the regression line with a custom label for the legend
        axes[i].plot(x_values, y_pred, color='red', label=legend_label)

        axes[i].set_title(f'{dep_var} vs {indep_var}')
        axes[i].set_xlabel(indep_var)
        axes[i].set_ylabel(dep_var)
        axes[i].legend()

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

    return models

def multivariate_regression_3d_plot(dep_var, indep_vars, data, show_plot = False, elevation=30, azimuth=-60, outliers_removed=False):
    # Create a copy of the original data to avoid modifying it directly
    df = data.copy()

    # Ensure indep_vars is a list of two variables
    if len(indep_vars) != 2:
        raise ValueError("This function requires exactly two independent variables.")

    if outliers_removed:
        # Function to remove outliers using the IQR method
        def remove_outliers(df, column):
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

        # Remove outliers for each variable in the model
        for column in [dep_var] + indep_vars:
            df = remove_outliers(df, column)

    # Create the formula string
    formula = f'{dep_var} ~ {" + ".join(indep_vars)}'

    # Fit the multivariate regression model
    model = ols(formula=formula, data=df).fit()

    if show_plot:
        # Extract R-squared value
        r_squared = model.rsquared

        # Create a meshgrid for the two independent variables
        x1_range = np.linspace(df[indep_vars[0]].min(), df[indep_vars[0]].max(), 10)
        x2_range = np.linspace(df[indep_vars[1]].min(), df[indep_vars[1]].max(), 10)
        X1, X2 = np.meshgrid(x1_range, x2_range)

        # Prepare the DataFrame for prediction
        X_grid = pd.DataFrame({
            indep_vars[0]: X1.ravel(),
            indep_vars[1]: X2.ravel()
        })

        # Predict the dependent variable for the meshgrid
        y_pred = model.predict(X_grid).values.reshape(X1.shape)

        # Plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Make the panes (subplot backgrounds) transparent
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        # Set the pane colors to transparent
        ax.xaxis.pane.set_edgecolor('gray')
        ax.yaxis.pane.set_edgecolor('gray')
        ax.zaxis.pane.set_edgecolor('gray')

        # Scatter plot of the actual data points
        ax.scatter(df[indep_vars[0]], df[indep_vars[1]], df[dep_var], color='blue', label='Actual Data')

        # Plot the regression plane
        ax.plot_surface(X1, X2, y_pred, alpha=0.5, color='red')

        ax.set_xlabel(indep_vars[0], labelpad=5)
        ax.set_ylabel(indep_vars[1], labelpad=5)
        ax.set_zlabel(dep_var, labelpad=10)
        ax.set_title('Multivariate Regression Analysis')

        # Set the viewing angle
        ax.view_init(elev=elevation, azim=azimuth)

        # Manually create a legend entry for the regression plane
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Actual Data', markerfacecolor='blue', markersize=10),
            Line2D([0], [0], color='red', label=f'Regression Plane (R² = {r_squared:.2f})', lw=2)
        ]

        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.35, 1))

        # Adjust layout to prevent labels from being cut off
        plt.tight_layout()

        plt.show()

    return model

def check_regression_assumptions(model_input, X, multi = False):
    columnx = model_input.model.exog
    fitted_values = model_input.fittedvalues

    # 1. Multicollinearity: Variance Inflation Factor (VIF)
    if multi:
        vif_data = pd.DataFrame()
        vif_data["feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
        print("Variance Inflation Factor (VIF) for Multicollinearity:")
        print(vif_data)
        print("VIF > 5 indicates high multicollinearity.")


    # 2. Homoscedasticity: Breusch-Pagan test
    residuals = model_input.resid
    bp_test = het_breuschpagan(residuals, columnx)
    bp_statistic, bp_p_value, f_statistic, f_p_value = bp_test
    print("\nBreusch-Pagan test for Homoscedasticity:")
    print(f"BP Statistic: {bp_statistic:.4f}, p-value: {bp_p_value:.4f}")
    if bp_p_value < 0.05:
        print("The residuals are heteroscedastic (reject the null hypothesis).")
    else:
        print("The residuals are homoscedastic (fail to reject the null hypothesis).")



    # Create a figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))

    # Plot residuals vs fitted values for Homoscedasticity
    sns.residplot(x=fitted_values, y=residuals, lowess=True, line_kws={'color': 'red', 'lw': 1}, ax=axes[0])
    axes[0].set_xlabel('Fitted values')
    axes[0].set_ylabel('Residuals')
    axes[0].set_title('Residuals vs Fitted Values')

    # Plot Q-Q plot of residuals for Normality
    sm.qqplot(residuals, line='s', ax=axes[1])
    axes[1].set_title("Normal Q-Q Plot of Residuals")

    plt.tight_layout()
    plt.show()

def evaluate_decision_tree_regressor(data, target_column='loyalty_points', train_proportion=0.7, use_columns=None, exclude_columns=None, depth=3, showplot=True, showstats=True):
    # Determine which columns to use as features
    if use_columns:
        X = data[use_columns].copy()
        columns_for_outliers = use_columns
    else:
        X = data.drop(columns=exclude_columns + [target_column] if exclude_columns else [target_column]).copy()
        columns_for_outliers = X.columns.tolist()

    y = data[target_column].copy()

    # Fill missing values
    for column in X.columns:
        if X[column].dtype == 'object':
            X[column].fillna(X[column].mode()[0], inplace=True)
        else:
            X[column].fillna(X[column].median(), inplace=True)

    # Encode categorical variables
    categorical_columns = X.select_dtypes(include=['object']).columns
    X = pd.get_dummies(X, columns=categorical_columns)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_proportion, random_state=42)

    # Create and fit the decision tree regressor model
    tree_regressor = DecisionTreeRegressor(max_depth=depth, random_state=42)
    tree_regressor.fit(X_train, y_train)

    # Predict the target variable for the test set
    y_pred = tree_regressor.predict(X_test)

    # Calculate performance metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    if showplot and depth < 8:
        # Plot the decision tree
        plt.figure(figsize=(20, 10))
        plot_tree(tree_regressor, feature_names=X.columns, filled=True, rounded=True, precision=3, impurity=True)
        plt.show()

    if showstats:
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"R-squared: {r2:.2f}")

    return {
        "mean_squared_error": mse,
        "r2_score": r2
    }

def evaluate_random_forest_regressor(data, target_column='loyalty_points', train_proportion=0.7, use_columns=None, exclude_columns=None, n_estimators=100, max_depth=3, showstats=True):
    # Determine which columns to use as features
    if use_columns:
        X = data[use_columns].copy()
    else:
        X = data.drop(columns=exclude_columns + [target_column] if exclude_columns else [target_column]).copy()

    y = data[target_column].copy()

    # Fill missing values
    for column in X.columns:
        if X[column].dtype == 'object':
            X[column].fillna(X[column].mode()[0], inplace=True)
        else:
            X[column].fillna(X[column].median(), inplace=True)

    # Encode categorical variables
    categorical_columns = X.select_dtypes(include=['object']).columns
    X = pd.get_dummies(X, columns=categorical_columns)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_proportion, random_state=42)

    # Create and fit the Random Forest regressor model
    rf_regressor = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    rf_regressor.fit(X_train, y_train)

    # Predict the target variable for the test set
    y_pred = rf_regressor.predict(X_test)

    # Calculate performance metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    if showstats:
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"R-squared: {r2:.2f}")

    return {
        "mean_squared_error": mse,
        "r2_score": r2
    }


def evaluate_model_with_seed(model, X, y, seed=42):
    # Split data using a different random seed
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

    # Fit model
    model.fit(X_train, y_train)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Scores
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)

    # Cross-validation on training set
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')

    # Report
    print(f"Random Seed: {seed}")
    print(f"Train R²: {train_r2:.3f}")
    print(f"Test R²:  {test_r2:.3f}")
    print(f"Train MSE: {train_mse:.3f}")
    print(f"Test MSE:  {test_mse:.3f}")
    print(f"CV R² Mean: {cv_scores.mean():.3f}")
    print(f"CV R² Std:  {cv_scores.std():.3f}")
    print("-" * 40)


# Example usage with a different seed
from sklearn.tree import DecisionTreeRegressor

best_params_bayesian = {
    'max_depth': 7,
    'min_samples_split': 66,
    'min_samples_leaf': 31,
    'ccp_alpha': 0.01,
    'min_impurity_decrease': 0.3
}

def format_as_table(df, text_column, polarity_columns):
    # Prepare the data for tabulate
    table_data = []
    headers = [text_column] + polarity_columns

    for index, row in df.iterrows():
        # Wrap the text
        wrapped_text = textwrap.fill(str(row[text_column]), width=80)
        # Format the polarities to three decimal places
        polarities = [f"{row[col]:.3f}" for col in polarity_columns]
        # Append the data to the table
        table_data.append([wrapped_text] + polarities)

    # Print the table
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

# Function to detokenize a list of tokens
def detokenize(tokens):
    return ' '.join(tokens)



# Initialize tools
lemmatizer = WordNetLemmatizer()
vader_analyzer = SentimentIntensityAnalyzer()
afinn = Afinn()
stop_words = set(stopwords.words('english'))

# Load RoBERTa sentiment model
roberta_model_name = "cardiffnlp/twitter-roberta-base-sentiment"
roberta_tokenizer = AutoTokenizer.from_pretrained(roberta_model_name)
roberta_model = AutoModelForSequenceClassification.from_pretrained(roberta_model_name)
roberta_pipeline = pipeline(
    "sentiment-analysis",
    model=roberta_model,
    tokenizer=roberta_tokenizer,
    truncation=True,
    max_length=512         # <-- max token length for RoBERTa
)

# Load BERT sentiment pipeline with truncation and max_length
bert_pipeline = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment",
    truncation=True,
    max_length=512          # <-- max token length for Bert
)

def expand_contractions(text):
    return contractions.fix(text)

def normalize_repeated_chars(text):
    # Replace 3 or more repeated characters with 2 chars ("soooo" -> "soo")
    return re.sub(r'(.)\1{2,}', r'\1\1', text)

def preprocess_for_textblob(text):
    # 1. Expand contractions
    text = expand_contractions(text)

    # 2. Lowercase
    text = text.lower()

    # 3. Remove URLs, mentions, hashtags
    text = re.sub(r"http\S+|www.\S+|@\w+|#\w+", "", text)

    # 4. Remove non-alphabetic characters (keep spaces)
    text = re.sub(r"[^a-z\s]", "", text)

    # 5. Normalize repeated characters
    text = normalize_repeated_chars(text)

    # 6. Tokenize
    words = word_tokenize(text)

    # 7. Lemmatize and remove stopwords
    lemmatized = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

    return " ".join(lemmatized)

def basic_preprocess(text):
    # For VADER, Afinn, RoBERTa — keep casing, punctuation, emojis

    # 1. Expand contractions to preserve meaning
    text = expand_contractions(text)

    # 2. Remove URLs, mentions, hashtags
    text = re.sub(r"http\S+|www.\S+|@\w+|#\w+", "", text)

    # 3. Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # 4. Normalize repeated characters
    text = normalize_repeated_chars(text)

    return text

def textblob_polarity(comment):
    if comment.strip():
        clean_comment = preprocess_for_textblob(comment)
        return TextBlob(clean_comment).sentiment.polarity
    return None

def vader_polarity(comment):
    if comment.strip():
        clean_comment = basic_preprocess(comment)
        return vader_analyzer.polarity_scores(clean_comment)['compound']
    return None

def afinn_polarity(comment):
    if comment.strip():
        clean_comment = basic_preprocess(comment)
        return afinn.score(clean_comment)
    return None

# Define length-normalized and clipped scaling function
def scale_afinn_by_length(text, score):
    if score is None or not text.strip():
        return None
    word_count = len(text.split())
    if word_count == 0:
        return 0
    normalized_score = score / word_count
    return max(min(normalized_score, 1), -1)

def roberta_scaled_scores(comments):
    filtered_comments = [c for c in comments if c.strip()]
    if not filtered_comments:
        return [None] * len(comments)

    cleaned = [basic_preprocess(c) for c in filtered_comments]
    results = roberta_pipeline(cleaned, truncation=True, max_length=512)

    label_map = {
        'LABEL_0': -1,  # negative
        'LABEL_1': 0,   # neutral
        'LABEL_2': +1   # positive
    }

    scaled_scores = []
    idx = 0
    for c in comments:
        if c.strip():
            r = results[idx]
            label = r['label']
            score = r['score']
            sentiment_value = label_map[label] * score
            scaled_scores.append(round(sentiment_value, 4))
            idx += 1
        else:
            scaled_scores.append(None)
    return scaled_scores

def bert_scaled_scores(comments):
    # Filter empty comments to avoid issues
    filtered_comments = [c for c in comments if c.strip()]
    if not filtered_comments:
        return [None] * len(comments)

    # Run batch inference with truncation
    results = bert_pipeline(filtered_comments, truncation=True, max_length=512)

    scaled_scores = []
    idx = 0
    for c in comments:
        if c.strip():
            r = results[idx]
            label = r['label']      # e.g., "4 stars"
            score = r['score']      # confidence score

            stars = int(label.split()[0])
            scaled = ((stars - 1) / 4) * 2 - 1  # Map stars 1-5 to -1 to +1
            weighted_score = scaled * score

            scaled_scores.append(round(weighted_score, 4))
            idx += 1
        else:
            scaled_scores.append(None)

    return scaled_scores

def combined_sentiment_zero_aware(row, fields):
    scores = [row[field] for field in fields]
    valid_scores = [s for s in scores if s is not None and not pd.isna(s)]

    if not valid_scores:
        return None

    non_zero_scores = [s for s in valid_scores if s != 0.0]

    if non_zero_scores:
        return round(sum(non_zero_scores) / len(non_zero_scores), 4)
    else:
        return 0.0  # only if all non-zero scores are truly zero

def get_best_rf_polarity(row):
    review_pol = row['combined_review_polarity']
    summary_pol = row['combined_summary_polarity']
    tag = row['rf_predicted_tag']

    same_sign = (review_pol > 0 and summary_pol > 0) or (review_pol < 0 and summary_pol < 0)
    opposite_sign = review_pol * summary_pol < 0
    diff_large = abs(review_pol - summary_pol) > 0.25

    if same_sign:
        return round((review_pol + summary_pol) / 2, 4)

    if opposite_sign:
        if diff_large:
            if tag == 'review':
                return round(review_pol, 4)
            elif tag == 'summary':
                return round(summary_pol, 4)
            elif tag == 'between':
                return round((review_pol + summary_pol) / 2, 4)
        else:
            return round((review_pol + summary_pol) / 2, 4)

    # fallback (e.g., one is 0)
    return round((review_pol + summary_pol) / 2, 4)

