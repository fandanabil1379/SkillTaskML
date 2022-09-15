import plot
import nn_model

import numpy as np

import mlflow
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

import sklearn.metrics as sm

class trainANN():
    def __init__(self, data, experiment_name=None, tags=None):
        self.tags = tags
        self.data = data
        
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment(experiment_name)
        self.experiment = mlflow.get_experiment_by_name(experiment_name)
        self.experimentID = self.experiment.experiment_id
    
    @staticmethod
    def score(y_actual, y_predict):
        metrics = {
            "Accuracy" : round(sm.accuracy_score(y_actual, y_predict), 3),
            "Recall" : round(sm.recall_score(y_actual, y_predict, average='weighted'), 3),
            "Precision" : round(sm.precision_score(y_actual, y_predict, average='weighted'), 3),
            "F1-Score" : round(sm.f1_score(y_actual, y_predict, average='weighted'), 3),
            "MCC" : round(sm.matthews_corrcoef(y_actual, y_predict), 3),
        }
        return metrics

    def run(self, n_iter=5, run_name='run@1', fitted_param=None, random=None):
        for iteration in range(n_iter):
            model = nn_model.baseModel()
            with mlflow.start_run(experiment_id=self.experimentID, run_name=run_name):
                data = nn_model.prepData(self.data, random_state=random)
                history = model.fit(data["Xtrain"], data["ytrain"], validation_data=(data["Xtest"], data["ytest"]), **fitted_param) 

                input_schema = Schema([
                    ColSpec("double", "sepal_length"),
                    ColSpec("double", "sepal_width"),
                    ColSpec("double", "petal_length"),
                    ColSpec("double", "petal_width"),
                ])
                output_schema = Schema([ColSpec("integer", "species")])
                signature = ModelSignature(inputs=input_schema, outputs=output_schema)
                mlflow.keras.log_model(model, "keras-ann-model", signature=signature)

                y_actual = np.argmax(data["ytest"], axis=1)
                y_predict = np.argmax(model.predict(data["Xtest"]), axis=1)
                mlflow.log_metrics(self.score(y_actual, y_predict))

                mlflow.log_figure(plot.cm_plot(y_actual, y_predict), "confusionMatrix.png")
                mlflow.log_figure(plot.accuracy_plot(history), "accuracyPlot.png")
                mlflow.log_figure(plot.loss_plot(history), "lossPlot.png")

                mlflow.set_tags(self.tags)
                mlflow.log_dict(data, "datasetTrainTest.json")
                mlflow.log_dict(history.history, "historyTrain.json")
                mlflow.log_params(fitted_param)  
                mlflow.log_params(model._get_compile_args())
                
        print("\nTraining has been completed, Log File Location: {}".format(self.experiment.artifact_location))
        
    def getBestModel(self, metrics):
        df = mlflow.search_runs(self.experimentID, output_format='pandas')
        run_id = df.loc[df[f'metrics.{metrics}'].idxmax()]['run_id']
        print(f'\nThe best model id: {run_id}')
        return run_id