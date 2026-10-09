import io
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    IsolationForest
)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    confusion_matrix,
    classification_report
)

warnings.filterwarnings("ignore")
# =====================================================
# 1. CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="JARVIS AI Command Center",
    page_icon="🤖",
    layout="wide"
)
st.title("🤖 JARVIS AI COMMAND CENTER")
st.caption(
    "Intelligent Data Automation | Data Science | "
    "Machine Learning | Deep Learning"
)

st.markdown("""
<style>
.block-container {
    padding-top: 1.5rem;
}
[data-testid="stMetric"] {
    border: 1px solid #394150;
    padding: 14px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# 2. SIDEBAR
# =====================================================

st.sidebar.title("⚙️ System Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload your dataset",
    type=["csv", "xlsx"]
)

if uploaded_file is None:
    st.info(
        "Upload a CSV or Excel dataset, or use the "
        "built-in sample dataset."
    )

    use_sample = st.button("Load Sample Dataset")

    if use_sample:
        st.session_state["use_sample"] = True

    if not st.session_state.get("use_sample", False):
        st.stop()


# =====================================================
# 3. LOAD DATA
# =====================================================

def load_data(file):
    if file.name.lower().endswith(".csv"):
        return pd.read_csv(file)

    return pd.read_excel(file)


def sample_data():
    rng = np.random.default_rng(42)
    n = 250

    hours = rng.uniform(1, 10, n)
    attendance = rng.uniform(50, 100, n)
    previous = rng.uniform(35, 95, n)

    marks = (
        3.8 * hours
        + 0.25 * attendance
        + 0.35 * previous
        + rng.normal(0, 5, n)
    )

    marks = np.clip(marks, 0, 100)

    return pd.DataFrame({
        "Study_Hours": hours,
        "Attendance": attendance,
        "Previous_Marks": previous,
        "Final_Marks": marks,
        "Department": rng.choice(
            ["CSE", "IT", "AI_ML"],
            n
        )
    })


try:
    if uploaded_file is not None:
        df_original = load_data(uploaded_file)
    else:
        df_original = sample_data()

except Exception as error:
    st.error(f"Unable to load dataset: {error}")
    st.stop()


if df_original.empty or len(df_original.columns) == 0:
    st.error("The dataset is empty.")
    st.stop()


# =====================================================
# 4. DATA CLEANING
# =====================================================

st.sidebar.subheader("Data Cleaning")

remove_duplicates = st.sidebar.checkbox(
    "Remove duplicate rows",
    value=True
)

clean_missing = st.sidebar.checkbox(
    "Fill missing values",
    value=True
)

df = df_original.copy()

df.columns = [
    str(column).strip()
    for column in df.columns
]

if remove_duplicates:
    df = df.drop_duplicates()

if clean_missing:
    for column in df.columns:
        if df[column].isna().any():

            if pd.api.types.is_numeric_dtype(df[column]):
                median = df[column].median()

                if pd.notna(median):
                    df[column] = df[column].fillna(median)

            else:
                modes = df[column].mode(dropna=True)

                if not modes.empty:
                    df[column] = df[column].fillna(
                        modes.iloc[0]
                    )


numeric_df = df.select_dtypes(include=np.number)
categorical_df = df.select_dtypes(exclude=np.number)


# =====================================================
# 5. HELPER FUNCTIONS
# =====================================================

def explain_numeric(column):
    values = pd.to_numeric(
        df[column],
        errors="coerce"
    ).dropna()

    if values.empty:
        return f"No usable numeric values found in {column}."

    mean_value = values.mean()
    median_value = values.median()
    std_value = values.std()
    minimum = values.min()
    maximum = values.max()

    skew = values.skew()

    if pd.isna(skew):
        shape = "The distribution shape is uncertain."
    elif skew > 0.5:
        shape = "The distribution is positively skewed."
    elif skew < -0.5:
        shape = "The distribution is negatively skewed."
    else:
        shape = "The distribution is approximately symmetric."

    return (
        f"Analysis of {column}: "
        f"mean = {mean_value:.2f}, "
        f"median = {median_value:.2f}, "
        f"standard deviation = {std_value:.2f}, "
        f"minimum = {minimum:.2f}, "
        f"maximum = {maximum:.2f}. "
        f"{shape}"
    )


def explain_correlation(a, b):
    valid = df[[a, b]].dropna()

    if len(valid) < 3:
        return "Not enough observations to assess correlation."

    correlation = valid[a].corr(valid[b])

    if pd.isna(correlation):
        return "Correlation cannot be calculated for these columns."

    strength = abs(correlation)

    if strength < 0.2:
        description = "very weak linear relationship"
    elif strength < 0.4:
        description = "weak linear relationship"
    elif strength < 0.7:
        description = "moderate linear relationship"
    else:
        description = "strong linear relationship"

    direction = (
        "positive"
        if correlation > 0
        else "negative"
        if correlation < 0
        else "neutral"
    )

    return (
        f"{a} and {b} have a {description}. "
        f"Pearson correlation = {correlation:.3f}. "
        f"The association is {direction}. "
        "Correlation does not establish causation."
    )


# =====================================================
# 6. DATA OVERVIEW
# =====================================================

st.header("📊 System Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Rows", len(df))
c2.metric("Columns", len(df.columns))
c3.metric(
    "Missing Values",
    int(df.isna().sum().sum())
)
c4.metric(
    "Duplicate Rows in Original",
    int(df_original.duplicated().sum())
)

with st.expander("View Dataset"):
    st.dataframe(df, use_container_width=True)

with st.expander("Column Information"):
    info = pd.DataFrame({
        "Data Type": df.dtypes.astype(str),
        "Missing Values": df.isna().sum(),
        "Unique Values": df.nunique()
    })

    st.dataframe(info, use_container_width=True)

st.download_button(
    "⬇️ Download Cleaned Dataset",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="cleaned_data.csv",
    mime="text/csv"
)


# =====================================================
# 7. DATA SCIENCE
# =====================================================

st.header("🧪 Data Science Laboratory")

if not numeric_df.empty:

    st.subheader("Statistical Summary")

    st.dataframe(
        numeric_df.describe().T,
        use_container_width=True
    )

    st.subheader("Distribution Analysis")

    selected_numeric = st.selectbox(
        "Select a numerical column",
        numeric_df.columns,
        key="distribution_column"
    )

    chart_type = st.selectbox(
        "Select visualization",
        [
            "Histogram",
            "Box Plot",
            "Violin Plot",
            "Density Histogram"
        ]
    )

    if chart_type == "Histogram":
        fig = px.histogram(
            df,
            x=selected_numeric,
            marginal="box",
            title=f"Distribution of {selected_numeric}"
        )

    elif chart_type == "Box Plot":
        fig = px.box(
            df,
            y=selected_numeric,
            points="outliers",
            title=f"Box Plot: {selected_numeric}"
        )

    elif chart_type == "Violin Plot":
        fig = px.violin(
            df,
            y=selected_numeric,
            box=True,
            points="outliers",
            title=f"Violin Plot: {selected_numeric}"
        )

    else:
        fig = px.histogram(
            df,
            x=selected_numeric,
            histnorm="probability density",
            title=f"Density Distribution: {selected_numeric}"
        )

    st.plotly_chart(fig, use_container_width=True)

    st.info(explain_numeric(selected_numeric))

    st.subheader("Correlation Heatmap")

    if numeric_df.shape[1] >= 2:
        corr = numeric_df.corr()

        fig = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Feature Correlation Matrix"
        )

        st.plotly_chart(fig, use_container_width=True)

        corr_pairs = (
            corr.where(
                np.triu(
                    np.ones(corr.shape),
                    k=1
                ).astype(bool)
            )
            .stack()
        )

        if not corr_pairs.empty:
            pair = corr_pairs.abs().idxmax()

            st.info(
                explain_correlation(pair[0], pair[1])
            )

    st.subheader("Scatter Plot")

    col_a, col_b = st.columns(2)

    x_axis = col_a.selectbox(
        "X-axis",
        numeric_df.columns,
        index=0,
        key="scatter_x"
    )

    y_index = 1 if len(numeric_df.columns) > 1 else 0

    y_axis = col_b.selectbox(
        "Y-axis",
        numeric_df.columns,
        index=y_index,
        key="scatter_y"
    )

    color_column = st.selectbox(
        "Color points by (optional)",
        ["None"] + list(categorical_df.columns),
        key="scatter_color"
    )

    fig = px.scatter(
        df,
        x=x_axis,
        y=y_axis,
        color=(
            None
            if color_column == "None"
            else color_column
        ),
        trendline="ols",
        title=f"{y_axis} vs {x_axis}"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info(explain_correlation(x_axis, y_axis))

    st.subheader("Feature Comparison")

    fig = px.scatter_matrix(
        df,
        dimensions=list(numeric_df.columns),
        title="Multivariate Feature Relationships"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Outlier Detection: IQR Method")

    outlier_rows = []

    for column in numeric_df.columns:
        values = numeric_df[column].dropna()

        if len(values) < 4:
            continue

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        count = int(
            ((values < lower) | (values > upper)).sum()
        )

        outlier_rows.append({
            "Column": column,
            "Lower Bound": lower,
            "Upper Bound": upper,
            "Potential Outliers": count
        })

    if outlier_rows:
        st.dataframe(
            pd.DataFrame(outlier_rows),
            use_container_width=True
        )

else:
    st.warning(
        "Numerical analysis requires at least one numerical column."
    )


# =====================================================
# 8. CATEGORICAL VISUALIZATION
# =====================================================

st.header("🗂️ Categorical Analysis")

if not categorical_df.empty:

    cat_column = st.selectbox(
        "Select a categorical column",
        categorical_df.columns
    )

    counts = (
        df[cat_column]
        .astype(str)
        .value_counts()
        .reset_index()
    )

    counts.columns = ["Category", "Count"]

    fig = px.bar(
        counts,
        x="Category",
        y="Count",
        title=f"Category Distribution: {cat_column}"
    )

    st.plotly_chart(fig, use_container_width=True)

    if len(counts) <= 8:
        fig = px.pie(
            counts,
            names="Category",
            values="Count",
            title=f"Category Composition: {cat_column}"
        )

        st.plotly_chart(fig, use_container_width=True)

    most_common = counts.iloc[0]

    st.info(
        f"The most frequent category in {cat_column} "
        f"is {most_common['Category']}, appearing "
        f"{most_common['Count']} times."
    )


# =====================================================
# 9. MACHINE LEARNING
# =====================================================

st.header("🤖 Machine Learning Laboratory")

ml_mode = st.selectbox(
    "Choose a machine learning task",
    [
        "Regression",
        "Classification",
        "Clustering",
        "Anomaly Detection"
    ]
)


def build_preprocessor(X):
    numerical = X.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical = X.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    transformers = []

    if numerical:
        transformers.append((
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            numerical
        ))

    if categorical:
        transformers.append((
            "categorical",
            Pipeline([
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent")
                ),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore")
                )
            ]),
            categorical
        ))

    if not transformers:
        raise ValueError("No usable features were found.")

    return ColumnTransformer(transformers)


if ml_mode in ["Regression", "Classification"]:

    target = st.selectbox(
        "Select target column",
        df.columns,
        key="ml_target"
    )

    test_percent = st.slider(
        "Test dataset percentage",
        10,
        40,
        20
    )

    if st.button("🚀 Train Model"):

        ml_df = df.dropna(subset=[target]).copy()

        X = ml_df.drop(columns=[target])
        y = ml_df[target]

        if len(ml_df) < 10:
            st.error("At least 10 rows are recommended.")
            st.stop()

        if X.empty or y.nunique() < 2:
            st.error(
                "Choose a target with at least two distinct values "
                "and one or more input features."
            )
            st.stop()

        if ml_mode == "Classification":
            if y.nunique() > 50:
                st.error(
                    "Too many target classes for this demonstration."
                )
                st.stop()

            if y.value_counts().min() < 2:
                st.error(
                    "Each class needs at least two observations."
                )
                st.stop()

            if y.nunique() / len(y) > 0.5:
                st.error(
                    "The target has too many distinct classes."
                )
                st.stop()

        try:
            stratify = (
                y
                if ml_mode == "Classification"
                else None
            )

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_percent / 100,
                random_state=42,
                stratify=stratify
            )

            preprocessor = build_preprocessor(X)

            if ml_mode == "Regression":
                estimator = RandomForestRegressor(
                    n_estimators=150,
                    random_state=42
                )
            else:
                estimator = RandomForestClassifier(
                    n_estimators=150,
                    random_state=42,
                    class_weight="balanced"
                )

            model = Pipeline([
                ("preprocessing", preprocessor),
                ("model", estimator)
            ])

            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            st.session_state["trained_model"] = model
            st.session_state["ml_features"] = list(X.columns)
            st.session_state["ml_target"] = target
            st.session_state["ml_task"] = ml_mode

            results = pd.DataFrame({
                "Actual": y_test.to_numpy(),
                "Predicted": predictions
            })

            st.subheader("Model Performance")

            if ml_mode == "Regression":

                mae = mean_absolute_error(
                    y_test,
                    predictions
                )

                r2 = r2_score(
                    y_test,
                    predictions
                )

                c1, c2 = st.columns(2)

                c1.metric(
                    "Mean Absolute Error",
                    f"{mae:.3f}"
                )

                c2.metric(
                    "R² Score",
                    f"{r2:.3f}"
                )

                st.write(
                    "MAE measures average absolute prediction error. "
                    "R² compares the model with a mean-based baseline; "
                    "it may be negative."
                )

                fig = px.scatter(
                    results,
                    x="Actual",
                    y="Predicted",
                    title="Actual vs Predicted"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            else:

                accuracy = accuracy_score(
                    y_test,
                    predictions
                )

                precision = precision_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0
                )

                recall = recall_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0
                )

                f1 = f1_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0
                )

                a, b, c, d = st.columns(4)

                a.metric("Accuracy", f"{accuracy:.3f}")
                b.metric("Precision", f"{precision:.3f}")
                c.metric("Recall", f"{recall:.3f}")
                d.metric("F1 Score", f"{f1:.3f}")

                labels = np.unique(
                    np.concatenate([
                        np.asarray(y_test),
                        np.asarray(predictions)
                    ])
                )

                cm = confusion_matrix(
                    y_test,
                    predictions,
                    labels=labels
                )

                fig = px.imshow(
                    cm,
                    x=[str(v) for v in labels],
                    y=[str(v) for v in labels],
                    text_auto=True,
                    title="Confusion Matrix",
                    labels={
                        "x": "Predicted",
                        "y": "Actual"
                    }
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                st.code(
                    classification_report(
                        y_test,predictions,zero_division=0
                    )
                )
                st.dataframe(results,use_container_width=True)
                st.download_button("Download Model Predictions",
                                   data = results.to_csv(index=False).encode("utf-8"),
                file_name="model_predictions.csv",mime="text/csv")
                
                st.success("Training and evaluation completed.")
        except Exception as error:
            st.error(f"Model training failed:{error}")

# =====================================================
# 10. CLUSTERING
# =====================================================

elif ml_mode == "Clustering":

    if numeric_df.shape[1] < 2:
        st.warning(
            "Clustering visualization requires two numerical columns."
        )

    else:
        features = st.multiselect(
            "Choose clustering features",
            numeric_df.columns,
            default=list(numeric_df.columns[:2])
        )

        k = st.slider(
            "Number of clusters",
            min_value=2,
            max_value=10,
            value=3
        )

        if st.button("Run Clustering"):

            if len(features) < 2:
                st.error(
                    "Choose at least two numerical features."
                )
                st.stop()

            try:
                X = df[features].copy()

                imputer = SimpleImputer(
                    strategy="median"
                )

                scaler = StandardScaler()

                X_scaled = scaler.fit_transform(
                    imputer.fit_transform(X)
                )

                if len(X_scaled) <= k:
                    st.error(
                        "Number of rows must exceed cluster count."
                    )
                    st.stop()

                model = KMeans(
                    n_clusters=k,
                    random_state=42,
                    n_init=10
                )

                labels = model.fit_predict(X_scaled)

                result = df.copy()
                result["Cluster"] = labels.astype(str)

                fig = px.scatter(
                    result,
                    x=features[0],
                    y=features[1],
                    color="Cluster",
                    title="K-Means Cluster Visualization"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                st.dataframe(
                    result,
                    use_container_width=True
                )

                st.download_button(
                    "Download Clustered Data",
                    data=result.to_csv(index=False).encode("utf-8"),
                    file_name="clustered_data.csv",
                    mime="text/csv"
                )

            except Exception as error:
                st.error(
                    f"Clustering failed: {error}"
                )


# =====================================================
# 11. ANOMALY DETECTION
# =====================================================

elif ml_mode == "Anomaly Detection":

    if numeric_df.shape[1] < 2:
        st.warning(
            "Anomaly detection requires at least two numeric columns."
        )

    else:
        contamination = st.slider(
            "Expected anomaly proportion",
            0.01,
            0.20,
            0.05
        )

        if st.button("Detect Anomalies"):

            X = numeric_df.copy()

            imputer = SimpleImputer(
                strategy="median"
            )

            scaler = StandardScaler()

            X_scaled = scaler.fit_transform(
                imputer.fit_transform(X)
            )

            model = IsolationForest(
                contamination=contamination,
                random_state=42
            )

            labels = model.fit_predict(X_scaled)

            result = df.copy()

            result["Anomaly_Status"] = np.where(
                labels == -1,
                "Potential Anomaly",
                "Normal"
            )

            st.dataframe(
                result,
                use_container_width=True
            )

            st.info(
                f"Flagged {int((labels == -1).sum())} "
                "observations as potential anomalies. "
                "These are candidates for investigation, not "
                "confirmed errors or fraud."
            )

            st.download_button(
                "Download Anomaly Results",
                data=result.to_csv(index=False).encode("utf-8"),
                file_name="anomaly_results.csv",
                mime="text/csv"
            )


# =====================================================
# 12. DEEP LEARNING
# =====================================================

st.header("🧠 Deep Learning Laboratory")

st.write(
    "Train a simple neural network for numerical regression "
    "or classification."
)

st.warning(
    "Neural networks generally need enough representative data. "
    "This demonstration uses a simple feed-forward network."
)

deep_target = st.selectbox(
    "Deep learning target",
    df.columns,
    key="deep_target"
)

deep_task = st.selectbox(
    "Deep learning task",
    ["Regression", "Classification"],
    key="deep_task"
)

if st.button("Train Neural Network"):

    try:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(42)

        deep_df = df.dropna(
            subset=[deep_target]
        ).copy()

        X = deep_df.drop(
            columns=[deep_target]
        )

        y = deep_df[deep_target]

        if len(deep_df) < 30:
            st.error(
                "Use at least 30 rows for this demonstration; "
                "more data is generally preferable."
            )
            st.stop()

        if X.empty or y.nunique() < 2:
            st.error(
                "Select a target with multiple values and "
                "at least one feature."
            )
            st.stop()

        if deep_task == "Classification":
            if y.nunique() > 50:
                st.error(
                    "This demonstration supports up to 50 classes."
                )
                st.stop()

            if y.value_counts().min() < 2:
                st.error(
                    "Each class needs at least two observations."
                )
                st.stop()

            if y.nunique() / len(y) > 0.5:
                st.error(
                    "The target has too many distinct classes."
                )
                st.stop()

        prep = build_preprocessor(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=(
                y
                if deep_task == "Classification"
                else None
            )
        )

        X_train_processed = prep.fit_transform(X_train)
        X_test_processed = prep.transform(X_test)

        if hasattr(X_train_processed, "toarray"):
            X_train_processed = X_train_processed.toarray()
            X_test_processed = X_test_processed.toarray()

        X_train_processed = np.asarray(
            X_train_processed,
            dtype=np.float32
        )

        X_test_processed = np.asarray(
            X_test_processed,
            dtype=np.float32
        )

        if deep_task == "Regression":

            y_train_array = np.asarray(
                y_train,
                dtype=np.float32
            )

            y_test_array = np.asarray(
                y_test,
                dtype=np.float32
            )

            if not np.isfinite(y_train_array).all():
                st.error(
                    "Regression targets must be finite numbers."
                )
                st.stop()

            output_units = 1
            output_activation = "linear"

            loss = "mse"

        else:

            from sklearn.preprocessing import LabelEncoder

            encoder = LabelEncoder()

            y_train_array = encoder.fit_transform(
                y_train.astype(str)
            )

            try:
                y_test_array = encoder.transform(
                    y_test.astype(str)
                )
            except ValueError:
                st.error(
                    "A test class is absent from the training "
                    "split. Try a larger dataset."
                )
                st.stop()

            classes = len(encoder.classes_)

            output_units = (
                1 if classes == 2 else classes
            )

            output_activation = (
                "sigmoid" if classes == 2 else "softmax"
            )

            loss = (
                "binary_crossentropy"
                if classes == 2
                else "sparse_categorical_crossentropy"
            )

        network = tf.keras.Sequential([
            tf.keras.layers.Input(
                shape=(X_train_processed.shape[1],)
            ),
            tf.keras.layers.Dense(
                64,
                activation="relu"
            ),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(
                32,
                activation="relu"
            ),
            tf.keras.layers.Dense(
                output_units,
                activation=output_activation
            )
        ])

        network.compile(
            optimizer="adam",
            loss=loss
        )

        history = network.fit(
            X_train_processed,
            y_train_array,
            epochs=30,
            batch_size=16,
            validation_split=0.2,
            verbose=0
        )

        st.subheader("Neural Network Training History")

        history_df = pd.DataFrame(history.history)

        fig = px.line(
            history_df,
            y=["loss", "val_loss"],
            title="Training and Validation Loss"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        raw_predictions = network.predict(
            X_test_processed,
            verbose=0
        )

        if deep_task == "Regression":

            predictions = raw_predictions.ravel()

            mae = mean_absolute_error(
                y_test_array,
                predictions
            )

            st.metric(
                "Neural Network MAE",
                f"{mae:.3f}"
            )

        else:

            if len(encoder.classes_) == 2:
                predicted_indices = (
                    raw_predictions.ravel() >= 0.5
                ).astype(int)
            else:
                predicted_indices = raw_predictions.argmax(
                    axis=1
                )

            predictions = encoder.inverse_transform(
                predicted_indices
            )

            accuracy = accuracy_score(
                y_test.astype(str),
                predictions
            )

            st.metric(
                "Neural Network Accuracy",
                f"{accuracy:.2%}"
            )

        st.success(
            "Neural network training completed."
        )

    except ImportError:
        st.error(
            "TensorFlow is not installed. Run: "
            "python -m pip install tensorflow"
        )

    except Exception as error:
        st.error(
            f"Deep learning failed: {error}"
        )


# =====================================================
# 13. JARVIS COMMAND CENTER
# =====================================================

st.header("🎙️ JARVIS Command Center")

st.write(
    'Try commands such as "explain dashboard", '
    '"show correlation", or "analyze Study_Hours".'
)

command = st.text_input(
    "Enter a Jarvis command",
    placeholder="Jarvis, explain this dashboard"
)

if st.button("Execute Command"):

    cmd = command.lower().strip()

    st.subheader("🤖 JARVIS Response")

    if not cmd:
        st.warning("Enter a command first.")

    elif any(
        word in cmd
        for word in ["hello", "hi jarvis", "hey jarvis"]
    ):

        st.success(
            "Hello! I am JARVIS, your data science assistant. "
            "I can summarize datasets, explain statistics, "
            "describe correlations, and explain model metrics."
        )

    elif any(
        phrase in cmd
        for phrase in [
            "explain dashboard",
            "explain everything",
            "dashboard summary",
            "summarize dashboard"
        ]
    ):

        st.write(
            f"""
            **Dataset Summary**

            - Rows: {len(df)}
            - Columns: {len(df.columns)}
            - Numeric columns: {len(numeric_df.columns)}
            - Categorical columns: {len(categorical_df.columns)}
            - Missing cells remaining: {int(df.isna().sum().sum())}
            - Duplicate rows remaining: {int(df.duplicated().sum())}

            **Interpretation**

            The dashboard summarizes the uploaded dataset,
            displays distributions and relationships,
            and provides machine learning and deep learning
            tools where applicable.

            Statistical relationships do not automatically
            establish causal relationships. Model performance
            must be assessed using held-out data.
            """
        )

    elif "correlation" in cmd:

        if numeric_df.shape[1] >= 2:

            corr = numeric_df.corr()

            pairs = (
                corr.where(
                    np.triu(
                        np.ones(corr.shape),
                        k=1
                    ).astype(bool)
                )
                .stack()
            )

            if not pairs.empty:

                pair = pairs.abs().idxmax()

                st.write(
                    explain_correlation(
                        pair[0],
                        pair[1]
                    )
                )

            else:
                st.info(
                    "No valid feature pairs were found."
                )

        else:
            st.info(
                "At least two numerical columns are required."
            )

    elif "missing" in cmd:

        missing = df.isna().sum()

        missing = missing[missing > 0]

        if missing.empty:
            st.success(
                "No missing values remain in the current dataset."
            )
        else:
            st.dataframe(
                missing.rename("Missing Values")
            )

    elif "outlier" in cmd:

        st.write(
            "Outliers are observations far from the central "
            "spread of a feature. Use the IQR table above to "
            "inspect potential outliers. Do not remove them "
            "automatically without checking their meaning."
        )

    elif "machine learning" in cmd or "model" in cmd:

        st.write(
            "Machine learning learns patterns from training data. "
            "Regression predicts numerical outcomes; classification "
            "predicts classes; clustering groups observations; "
            "anomaly detection flags unusual records. "
            "Evaluate results on held-out data."
        )

    elif "deep learning" in cmd or "neural network" in cmd:

        st.write(
            "Deep learning uses neural networks with layers of "
            "learnable parameters. This dashboard demonstrates "
            "a feed-forward network using dense layers, ReLU "
            "activations, dropout, and the Adam optimizer."
        )

    elif "graph" in cmd or "visualization" in cmd:

        st.write(
            "Histogram: distribution of values. "
            "Box plot: median, quartiles, and potential outliers. "
            "Scatter plot: relationship between two variables. "
            "Heatmap: pairwise correlation. "
            "Bar chart: category counts. "
            "Confusion matrix: actual versus predicted classes."
        )

    elif "help" in cmd or "commands" in cmd:

        st.write("""
        Available commands:

        - `explain dashboard`
        - `show correlation`
        - `show missing values`
        - `explain outliers`
        - `explain machine learning`
        - `explain deep learning`
        - `explain graphs`
        - `analyze Study_Hours`
        """)

    else:

        matched_column = next(
            (
                col for col in df.columns
                if col.lower() in cmd
            ),
            None
        )

        if matched_column in numeric_df.columns:

            st.write(
                explain_numeric(matched_column)
            )

        elif matched_column is not None:

            counts = (
                df[matched_column]
                .astype(str)
                .value_counts()
                .head(10)
            )

            st.write(
                f"Category analysis for {matched_column}"
            )

            st.dataframe(
                counts.rename("Count")
            )

        else:

            st.warning(
                "Command not recognized. Type 'help' "
                "to see supported commands."
            )


# =====================================================
# 14. AUTOMATED INSIGHTS
# =====================================================

st.header("💡 Automated Data Insights")

if not numeric_df.empty:

    insights = []

    insights.append(
        f"The dataset contains {len(df)} rows and "
        f"{len(df.columns)} columns."
    )

    if len(numeric_df.columns) >= 2:

        corr = numeric_df.corr()

        pairs = (
            corr.where(
                np.triu(
                    np.ones(corr.shape),
                    k=1
                ).astype(bool)
            )
            .stack()
        )

        if not pairs.empty:

            pair = pairs.abs().idxmax()

            insights.append(
                explain_correlation(
                    pair[0],
                    pair[1]
                )
            )

    for column in numeric_df.columns[:5]:

        values = numeric_df[column].dropna()

        if not values.empty:

            insights.append(
                f"{column}: mean {values.mean():.2f}; "
                f"median {values.median():.2f}; "
                f"range {values.min():.2f} to "
                f"{values.max():.2f}."
            )

    for index, insight in enumerate(insights, start=1):

        st.write(f"**Insight {index}:** {insight}")


# =====================================================
# 15. FOOTER
# =====================================================

st.divider()

st.caption(
    "JARVIS AI | Data Science + Machine Learning + "
    "Deep Learning | Educational Project"
)