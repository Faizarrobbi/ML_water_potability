import os
import sys
import joblib
import argparse
import warnings

import mlflow
import yaml 
import numpy as np
import lightgbm as lgbm

from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.experimental import enable_iterative_imputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.impute import KNNImputer, SimpleImputer, IterativeImputer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    Normalizer,
    PolynomialFeatures,
    PowerTransformer,
    RobustScaler,
)

from data_utils import WaterPotabilityDataLoader

warnings.filterwarnings("ignore")


def load_mlflow_model(dir_mlflow_model):
    model_pipeline = mlflow.sklearn.load_model(dir_mlflow_model)
    return model_pipeline


class ClassificationPipeline:
    def __init__(self):
        self._imputer = None
        self._imputer_params = None

        self._preprocessor = None
        self._preprocessor_params = None

        self._transformer = None

        self._pca = None
        self._pca_params = None

        self._classifier = None
        self._classifier_params = None

        self.clf_pipeline = None
        self.clf_pipeline_params = None

    def set_imputer(self, imputer_type):

        if imputer_type == "simple":
            imputer = SimpleImputer()
            imputer_params = {
                "imputer__strategy": ["mean", "median"],
            }

        elif imputer_type == "knn":
            imputer = KNNImputer()
            imputer_params = {
                "imputer__n_neighbors": [5, 7],
                "imputer__weights": ["uniform", "distance"],
            }

        elif imputer_type == "iterative":
            imputer = IterativeImputer(random_state=42)
            imputer_params = {
                "imputer__initial_strategy": ["mean", "median"],
            }

        else:
            print(f"Unknown imputer type: {imputer_type}")
            sys.exit(0)

        self._imputer = imputer
        self._imputer_params = imputer_params

    def set_preprocessor(self, preprocessor_type):

        if preprocessor_type == "std":
            preprocessor = StandardScaler()
            preprocessor_params = None

        elif preprocessor_type == "min_max":
            preprocessor = MinMaxScaler(feature_range=(1, 2), clip=True)
            preprocessor_params = None

        elif preprocessor_type == "norm":
            preprocessor = Normalizer()
            preprocessor_params = {
                "preprocessor__norm": ["l1", "l2"],
            }

        elif preprocessor_type == "poly":
            preprocessor = PolynomialFeatures()
            preprocessor_params = {
                "preprocessor__degree": [2],
                "preprocessor__interaction_only": [True, False],
                "preprocessor__include_bias": [False],
            }

        elif preprocessor_type == "robust":
            preprocessor = RobustScaler()
            preprocessor_params = None

        else:
            print(f"Unknown preprocessor type: {preprocessor_type}")
            sys.exit(0)

        self._preprocessor = preprocessor
        self._preprocessor_params = preprocessor_params

    def set_transformer(self, transformer_type):

        if transformer_type == "power_box_cox":
            self._transformer = PowerTransformer(method="box-cox")

        elif transformer_type == "power_yeo_johnson":
            self._transformer = PowerTransformer(method="yeo-johnson")

        else:
            print(f"Unknown transformer type: {transformer_type}")
            sys.exit(0)

    def set_pca(self, max_num_feats):

        self._pca = PCA()

        self._pca_params = {
            "pca__n_components": np.arange(2, max_num_feats + 1),
        }

    def set_classifier(self, classifier_type):

        if classifier_type == "ada_boost":

            classifier = AdaBoostClassifier(
                algorithm="SAMME",
                random_state=42,
            )

            classifier_params = {
                "classifier__learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0],
                "classifier__n_estimators": [50, 100, 200],
            }

        elif classifier_type == "log_reg":

            classifier = LogisticRegression(
                max_iter=500,
                solver="saga",
                random_state=42,
            )

            classifier_params = {
                "classifier__penalty": ["l1", "l2"],
                "classifier__class_weight": [None, "balanced"],
                "classifier__C": [0.1, 0.5, 1, 2],
            }

        elif classifier_type == "random_forest":

            classifier = RandomForestClassifier(
                random_state=42,
                n_jobs=-1,
            )

            classifier_params = {
                "classifier__n_estimators": [100, 200],
                "classifier__criterion": ["gini", "entropy"],
                "classifier__max_depth": [None, 10, 25],
                "classifier__min_samples_leaf": [1, 5],
                "classifier__min_samples_split": [2, 5],
            }

        elif classifier_type == "svc":

            classifier = SVC()

            classifier_params = {
                "classifier__C": [0.5, 1, 2],
                "classifier__kernel": ["linear", "rbf"],
            }

        elif classifier_type == "light_gbm":

            classifier = lgbm.LGBMClassifier(
                boosting_type="gbdt",
                objective="binary",
                verbosity=-1,
                random_state=42,
            )

            classifier_params = {
                "classifier__num_leaves": [31, 63],
                "classifier__learning_rate": [0.01, 0.05, 0.1],
                "classifier__n_estimators": [100, 300],
                "classifier__reg_lambda": [0.1, 1.0],
                "classifier__min_data_in_leaf": [20, 50],
            }

        else:
            print(f"Unknown classifier type: {classifier_type}")
            sys.exit(0)

        self._classifier = classifier
        self._classifier_params = classifier_params

    def build_pipeline(self):

        pipeline_steps = [("imputer", self._imputer)]

        if self._preprocessor is not None:
            pipeline_steps.append(("preprocessor", self._preprocessor))

        if self._transformer is not None:
            pipeline_steps.append(("transformer", self._transformer))

        if self._pca is not None:
            pipeline_steps.append(("pca", self._pca))

        pipeline_steps.append(("classifier", self._classifier))

        self.clf_pipeline = Pipeline(pipeline_steps)

        list_pipeline_params = [self._imputer_params]

        if self._preprocessor_params is not None:
            list_pipeline_params.append(self._preprocessor_params)

        if self._pca_params is not None:
            list_pipeline_params.append(self._pca_params)

        list_pipeline_params.append(self._classifier_params)

        self._set_pipeline_params(list_pipeline_params)

    def _set_pipeline_params(self, list_pipeline_params):

        final_pipeline_params = {}

        for params in list_pipeline_params:
            final_pipeline_params.update(params)

        self.clf_pipeline_params = final_pipeline_params


def train_model(
    water_pot_dataset,
    imputer_type,
    preprocessor_type,
    transformer_type,
    classifier_type,
    is_pca=False,
):

    X_train, Y_train = water_pot_dataset.get_data_from_data_frame(
        which_set="train"
    )

    X_test, Y_test = water_pot_dataset.get_data_from_data_frame(
        which_set="test"
    )

    pca_str = "no_pca"
    preprocessor_str = "no_preproc"
    transformer_str = "no_transform"

    clf_pipeline = ClassificationPipeline()

    clf_pipeline.set_imputer(imputer_type)

    if preprocessor_type != "none":
        clf_pipeline.set_preprocessor(preprocessor_type)
        preprocessor_str = preprocessor_type

    if transformer_type != "none":

        transformer_str = transformer_type

        if transformer_type == "power_box_cox":
            clf_pipeline.set_preprocessor("min_max")
            clf_pipeline.set_transformer(transformer_type)
            preprocessor_str = "min_max"

        else:
            clf_pipeline.set_preprocessor("std")
            clf_pipeline.set_transformer(transformer_type)
            preprocessor_str = "std"

    if is_pca:
        clf_pipeline.set_pca(X_train.shape[1])
        pca_str = "pca"

    clf_pipeline.set_classifier(classifier_type)

    print("\n" + "-" * 100)
    clf_pipeline.build_pipeline()

    print(clf_pipeline.clf_pipeline)

    print("\n" + "-" * 100)
    print("Model pipeline params space:")
    print(clf_pipeline.clf_pipeline_params)
    print("-" * 100)

    k_fold_cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    grid_cv = GridSearchCV(
        clf_pipeline.clf_pipeline,
        clf_pipeline.clf_pipeline_params,
        scoring="f1",
        cv=k_fold_cv,
        n_jobs=-1,
        verbose=1,
    )

    grid_cv.fit(X_train, Y_train)

    cv_best_estimator = grid_cv.best_estimator_
    cv_best_f1 = grid_cv.best_score_
    cv_best_params = grid_cv.best_params_

    Y_train_pred = cv_best_estimator.predict(X_train)

    train_f1 = f1_score(Y_train, Y_train_pred)
    train_acc = accuracy_score(Y_train, Y_train_pred)

    Y_test_pred = cv_best_estimator.predict(X_test)

    test_f1 = f1_score(Y_test, Y_test_pred)
    test_acc = accuracy_score(Y_test, Y_test_pred)

    print("\n" + "-" * 50)
    print("Best Parameters:")
    print(cv_best_params)

    print("\nTrain Metrics")
    print(f"F1 Score : {train_f1:.4f}")
    print(f"Accuracy : {train_acc:.4f}")

    print("\nTest Metrics")
    print(f"F1 Score : {test_f1:.4f}")
    print(f"Accuracy : {test_acc:.4f}")

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("water_potability")

    experiment = mlflow.get_experiment_by_name(
        "water_potability"
    )

    print("\nStarted mlflow logging...")

    model_log_str = (
        f"{imputer_type}_"
        f"{preprocessor_str}_"
        f"{transformer_str}_"
        f"{pca_str}_"
        f"{classifier_type}"
    )

    with mlflow.start_run(
        experiment_id=experiment.experiment_id
    ):

        mlflow.sklearn.log_model(
    sk_model=cv_best_estimator,
    artifact_path="model",
    registered_model_name="water_potability",
)

        mlflow.log_params(cv_best_params)

        mlflow.log_metric("cv_f1_score", cv_best_f1)
        mlflow.log_metric("train_f1_score", train_f1)
        mlflow.log_metric("train_acc_score", train_acc)
        mlflow.log_metric("test_f1_score", test_f1)
        mlflow.log_metric("test_acc_score", test_acc)

    mlflow.end_run()

    print("Completed mlflow logging")
    print("-" * 50)


def init_and_train_model(ARGS):

    water_pot_dataset = WaterPotabilityDataLoader(
        ARGS.file_csv
    )

    water_pot_dataset.read_csv_file()
    water_pot_dataset.split_data()

    num_samples_train = water_pot_dataset.df_train.shape[0]
    num_samples_test = water_pot_dataset.df_test.shape[0]

    print("\n" + "-" * 40)
    print("Num samples after splitting the dataset")
    print("-" * 40)

    print(
        f"train: {num_samples_train}, "
        f"test: {num_samples_test}"
    )

    print("\n" + "-" * 40)
    print("A few samples from train data")
    print("-" * 40)

    print(water_pot_dataset.df_train.head())

    if ARGS.is_train:

        train_model(
            water_pot_dataset,
            ARGS.imputer_type,
            ARGS.preprocessor_type,
            ARGS.transformer_type,
            ARGS.classifier_type,
            bool(ARGS.is_pca),
        )

def load_config():
    with open("config/train_config.yaml", "r") as f:
        return yaml.safe_load(f)

def main():

    config = load_config()

    class Args:
        pass

    args = Args()

    args.file_csv = config["file_csv"]
    args.classifier_type = config["classifier_type"]
    args.imputer_type = config["imputer_type"]
    args.preprocessor_type = config["preprocessor_type"]
    args.transformer_type = config["transformer_type"]
    args.is_pca = int(config["is_pca"])
    args.is_train = 1

    init_and_train_model(args)


if __name__ == "__main__":
    main()