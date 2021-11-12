# other modules
import numpy as np
import optuna
# Regression models
import lightgbm as lgb
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
# Classification models
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
# Classification metrics
from sklearn.metrics import accuracy_score, precision_score, log_loss, recall_score, f1_score
# Regression metrics
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

store_score = []
models_list = []
reg_models = ["RandomForestRegressor", "ExtraTreesRegressor", "LinearRegression", "CatBoostRegressor", "LGBMRegressor"]
cla_models = ["RandomForestClassifier", "ExtraTreesClassifier", "CatBoostClassifier", "LGBMClassifier", "LogisticRegression"]


class autoML():

    def train_model(self, model, metric, X_train, X_val, y_train, y_val):
        def objective(trial):
            try:
                # regression
                if (model == "RandomForestRegressor" or model == "ExtraTreesRegressor"):
                    n_estimators = trial.suggest_int("n_estimators", 1, 100)
                    max_depth = trial.suggest_int("max_depth", 2, 15)
                    if (model == "RandomForestRegressor"):
                        rfr = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, bootstrap=True)
                        rfr.fit(X_train, y_train)
                        y_pred = rfr.predict(X_val)
                        models_list.append(rfr)
                    else:
                        etr = ExtraTreesRegressor(n_estimators=n_estimators, max_depth=max_depth, bootstrap=True)
                        etr.fit(X_train, y_train)
                        y_pred = etr.predict(X_val)
                        models_list.append(etr)

                elif (model == "LinearRegression"):
                    lr = LinearRegression()
                    lr.fit(X_train, y_train)
                    y_pred = lr.predict(X_val)
                    models_list.append(lr)

                elif (model == "CatBoostRegressor"):
                    learning_rate = trial.suggest_discrete_uniform("learning_rate", 0.01, 0.09, 0.01)
                    depth = trial.suggest_int("depth", 2, 15)
                    n_estimators = trial.suggest_int("n_estimators", 1, 100)
                    cbr = CatBoostRegressor(learning_rate=learning_rate, depth=depth, logging_level="Silent",
                                            n_estimators=n_estimators)
                    cbr.fit(X_train, y_train)
                    y_pred = cbr.predict(X_val)
                    models_list.append(cbr)

                elif (model == "LGBMRegressor"):
                    learning_rate = trial.suggest_discrete_uniform("learning_rate", 0.01, 0.09, 0.01)
                    n_estimators = trial.suggest_int("n_estimators", 1, 100)
                    lgbr = lgb.LGBMRegressor(silent=True, n_estimators=n_estimators, learning_rate=learning_rate)
                    lgbr.fit(X_train, y_train)
                    y_pred = lgbr.predict(X_val)
                    models_list.append(lgbr)


                # classification
                elif (model == "RandomForestClassifier" or model == "ExtraTreesClassifier"):
                    n_estimators = trial.suggest_int("n_estimators", 1, 100)
                    criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])
                    max_depth = trial.suggest_int("max_depth", 2, 15)
                    if (model == "RandomForestClassifier"):
                        rfc = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion,
                                                     max_depth=max_depth)
                        rfc.fit(X_train, y_train)
                        y_pred = rfc.predict(X_val)
                        models_list.append(rfc)
                    else:
                        etc = ExtraTreesClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth)
                        etc = etc.fit(X_train, y_train)
                        y_pred = etc.predict(X_val)
                        models_list.append(etc)

                elif (model == "CatBoostClassifier"):
                    learning_rate = trial.suggest_discrete_uniform("learning_rate", 0.01, 0.09, 0.01)
                    depth = trial.suggest_int("depth", 2, 15)
                    n_estimators = trial.suggest_int("n_estimators", 1, 100)
                    cbc = CatBoostClassifier(learning_rate=learning_rate, depth=depth, logging_level="Silent",
                                             n_estimators=n_estimators)
                    cbc.fit(X_train, y_train)
                    y_pred = cbc.predict(X_val)
                    models_list.append(cbc)

                elif (model == "LGBMClassifier"):
                    learning_rate = trial.suggest_discrete_uniform("learning_rate", 0.01, 0.09, 0.01)
                    n_estimators = trial.suggest_int("n_estimators", 1, 100)
                    lgbr = lgb.LGBMClassifier(silent=True, n_estimators=n_estimators, learning_rate=learning_rate)
                    lgbr.fit(X_train, y_train)
                    y_pred = lgbr.predict(X_val)
                    models_list.append(lgbr)

                elif (model == "LogisticRegression"):
                    logr = LogisticRegression()
                    logr.fit(X_train, y_train)
                    y_pred = logr.predict(X_val)
                    models_list.append(logr)

            except Exception as e:
                print("Something went wrong with " + model)
                return e

            try:
                # scores
                if (metric == "mae"):
                    score = mean_absolute_error(y_val, y_pred)

                elif (metric == "mse"):
                    score = mean_squared_error(y_val, y_pred)

                elif (metric == "r2_score"):
                    score = r2_score(y_val, y_pred)

                elif (metric == "rmse"):
                    mse_ = mean_squared_error(y_val, y_pred)
                    score = np.sqrt(mse_)

                elif (metric == "accuracy_score"):
                    score = accuracy_score(y_val, y_pred)

                elif (metric == "log_loss"):
                    score = log_loss(y_val, y_pred)

                elif (metric == "precision_score"):
                    score = precision_score(y_val, y_pred)

                elif (metric == "recall_score"):
                    score = recall_score(y_val, y_pred)

                elif (metric == "f1_score"):
                    score = f1_score(y_val, y_pred)

                else:
                    score = 0

                return score

            except Exception as e:
                score = 0
                return e


        optuna.logging.disable_default_handler()
        if metric_ == "mae" or metric_ == "mse" or metric_ == "rmse" or metric_ == "log_loss":
            study = optuna.create_study(direction="minimize", pruner=optuna.pruners.SuccessiveHalvingPruner())

        elif metric_ == "r2_score" or metric_ == "accuracy_score" or metric_ == "f1_score" or metric_ == "recall_score" or metric_ == "precision_score":
            study = optuna.create_study(direction="maximize", pruner=optuna.pruners.SuccessiveHalvingPruner())

        study.optimize(objective, n_trials=10)
        store_score.append(study.best_value)

        print("model name: ", model, " Best params: ", study.best_params, " Score: ", study.best_value)

    def fit(self, X_train, X_val, y_train, y_val, task, metric):

        store_score.clear()
        models_list.clear()

        global metric_
        metric_ = metric

        if task == "Regression" or task == "regression":

             self.train_model("RandomForestRegressor", metric, X_train, X_val, y_train, y_val)

             self.train_model("ExtraTreesRegressor", metric, X_train, X_val, y_train, y_val)

             self.train_model("LinearRegression", metric, X_train, X_val, y_train, y_val)

             self.train_model("CatBoostRegressor", metric, X_train, X_val, y_train, y_val)

             self.train_model("LGBMRegressor", metric, X_train, X_val, y_train, y_val)

        elif task == "Classification" or task == "classification":

            self.train_model("RandomForestClassifier", metric, X_train, X_val, y_train, y_val)

            self.train_model("ExtraTreesClassifier", metric, X_train, X_val, y_train, y_val)

            self.train_model("CatBoostClassifier", metric, X_train, X_val, y_train, y_val)

            self.train_model("LGBMClassifier", metric, X_train, X_val, y_train, y_val)

            self.train_model("LogisticRegression", metric, X_train, X_val, y_train, y_val)

        else:
            print("task can either be regression or classification")

    def get_best_model(self):
        try:

            if metric_ == "mae" or metric_ == "mse" or metric_ == "rmse":
                min_val = min(store_score)
                indx = store_score.index(min_val)

                return reg_models[indx]

            elif metric_ == "r2_score":
                max_val = max(store_score)
                indx = store_score.index(max_val)

                return reg_models[indx]

            elif metric_ == "accuracy_score" or metric_ == "f1_score" or metric_ == "recall_score" or metric_ == "precision_score":
                max_val = max(store_score)
                indx = store_score.index(max_val)

                return cla_models[indx]

            elif metric_ == "log_loss":
                min_val = min(store_score)
                indx = store_score.index(min_val)

                return cla_models[indx]

        except Exception as e:
            return e


    def predict(self,X_val):
        try:
            if autoML().get_best_model() == "RandomForestRegressor" or autoML().get_best_model() == "RandomForestClassifier":
                return models_list[0].predict(X_val)

            elif autoML().get_best_model() == "ExtraTreesRegressor" or autoML().get_best_model() == "ExtraTreesClassifier":
                return models_list[1].predict(X_val)

            elif autoML().get_best_model() == "LinearRegression" or autoML().get_best_model() == "CatBoostClassifier":
                return models_list[2].predict(X_val)

            elif autoML().get_best_model() == "CatBoostRegressor" or autoML().get_best_model() == "LGBMClassifier":
                return models_list[3].predict(X_val)

            elif autoML().get_best_model() == "LGBMRegressor" or autoML().get_best_model() == "LogisticRegression":
                return models_list[4].predict(X_val)

        except Exception as e:
            return e
