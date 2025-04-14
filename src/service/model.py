import logging
import pandas as pd
from typing import Optional, Tuple, Dict, Literal, Any
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from src.enums import Feature
from src.repository.transaction_repo import get_multiple_rows
from src.entity.model import Model

logger = logging.getLogger(__name__)

def preprocess_data(df: pd.DataFrame, test_size: float = 0.2) -> Tuple:
    """
    Split the dataframe into X (features) and y (target).

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        Tuple: Split data (X_train, X_test, y_train, y_test).
    """
    # Format data for the model
    df_model = df

    features = [f.name for f in Feature.get_all_features()]
    X = df_model[features]
    y = df_model['is_fraud']

    # Split the data
    if test_size > 0:
        return train_test_split(X, y, test_size=test_size, stratify=df_model.is_fraud, random_state=42)
    else:
        return X, pd.DataFrame(), y, pd.DataFrame()

def train_model(
        pipeline: Pipeline,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame) -> Pipeline:
    """
    Train the pipeline
    """
    # Start the timer
    import time
    start_time = time.time()
    print("Training the model...")
    pipeline.fit(X_train, y_train)
    print(f"Model trained in {time.time() - start_time:.2f} seconds")
    return pipeline

all_algorithms = Literal[
    'XGBoost',
    'RandomForest',
    'SVM',
    'GradientBoosting',
    'MLP',
    'LightGBM',
    'XGBRF',
    'DecisionTree',
    'ExtraTrees',
    'Bagging',
]

def create_and_train_model(data: pd.DataFrame,
                           evaluate: bool = False,
                           algo: all_algorithms = 'MLP') -> Pipeline:
    """
    Create and train a model on the given data
    """
    if evaluate:
        test_size = 0.2
    else:
        test_size = 0.0
    
    # Split the data
    X_train, X_test, y_train, y_test = preprocess_data(df=data, test_size=test_size)

    # Train the model
    pipeline = create_pipeline(algo)
    pipeline = train_model(pipeline, X_train, y_train)

    if evaluate:
        evaluation_results = evaluate_model(pipeline, X_test, y_test)
        logging.info(f"Evaluation results for {algo}:")
        logging.info(f"F1 Score: {evaluation_results['f1_score']}\n")
        logging.info(f"Confusion Matrix:\n{evaluation_results['confusion_matrix']}\n")

    return pipeline

def evaluate_model(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
    """
    Evaluates the model
    """
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    return {
        "accuracy": accuracy,
        "f1_score": f1,
        "confusion_matrix": cm
    }

def train_model_from_scratch(limit: Optional[int] = None,
                             evaluate: bool = False,
                             algo: all_algorithms = 'MLP',
                             output_path: str = './data/model.pkl') -> Pipeline:
    """
    Train a model from scratch
    """
    # Load data
    data = get_multiple_rows(limit=limit)

    # Train the model
    pipeline = create_and_train_model(data=data, evaluate=evaluate, algo=algo)

    # Save the model
    metadata = {
        "model_name": algo,
        "version": "1.0",
        "training_data_size": len(data),
        "training_datetime": pd.Timestamp.now().isoformat(),
    }
    Model.save_model(pipeline, metadata, output_path)

    return pipeline

def create_pipeline(algo: all_algorithms = 'XGBoost') -> Pipeline:
    """
    Creates a machine learning pipeline.

    Returns:
        Pipeline: A scikit-learn pipeline object.
    """
    # Define the features, numerical and categorical
    cat_features = [f.name for f in Feature.get_features_by_type('category')]
    num_features = [f.name for f in Feature.get_features_by_type('number')]

    # Pipeline for numerical variables
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Pipeline for categorical variables
    cat_transformer = OneHotEncoder(handle_unknown='ignore')

    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_features),
            ('cat', cat_transformer, cat_features)
        ]
    )

    # Choose the classifier based on the algorithm
    if algo == 'XGBoost':
        from xgboost import XGBClassifier
        classifier = XGBClassifier(eval_metric='logloss')
    elif algo == 'RandomForest':
        from sklearn.ensemble import RandomForestClassifier
        classifier = RandomForestClassifier()
    elif algo == 'SVM':
        from sklearn.svm import SVC
        classifier = SVC(probability=False)
    elif algo == 'GradientBoosting':
        from sklearn.ensemble import GradientBoostingClassifier
        classifier = GradientBoostingClassifier()
    elif algo == 'MLP':
        from sklearn.neural_network import MLPClassifier
        classifier = MLPClassifier(max_iter=1000, verbose=True)
    elif algo == 'LightGBM':
        from lightgbm import LGBMClassifier
        classifier = LGBMClassifier()
    elif algo == 'XGBRF':
        from xgboost import XGBRFClassifier
        classifier = XGBRFClassifier(eval_metric='logloss')
    elif algo == 'DecisionTree':
        from sklearn.tree import DecisionTreeClassifier
        classifier = DecisionTreeClassifier()
    elif algo == 'ExtraTrees':
        from sklearn.ensemble import ExtraTreesClassifier
        classifier = ExtraTreesClassifier()
    elif algo == 'Bagging':
        from sklearn.ensemble import BaggingClassifier
        classifier = BaggingClassifier()
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    # Full pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])

    return pipeline

def predict(
    pipeline: Pipeline,
    job: str,
    city: str,
    state: str,
    category: str,
    amt: float,
    city_pop: int
) -> Dict[str, Any]:
    # Built a DataFrame with the new match
    transaction = pd.DataFrame([{
        Feature.CUSTOMER_CITY.name: city,
        Feature.CUSTOMER_CITY_POP.name: city_pop,
        Feature.CUSTOMER_JOB.name: job,
        Feature.CUSTOMER_STATE.name: state,
        Feature.TRANSACTION_AMOUNT.name: amt,
        Feature.TRANSACTION_CATEGORY.name: category
    }])

    # Use the pipeline to make a prediction
    prediction = pipeline.predict(transaction)[0]
    proba = pipeline.predict_proba(transaction)[0]

    # Print the result
    logging.info(f"Is fraud: {'True' if prediction == 1 else 'False'}")
    print(f"Probability of fraud: {proba}")
    # Return the result
    return {"result": prediction.item(), "fraud_probability": proba[1].item()}