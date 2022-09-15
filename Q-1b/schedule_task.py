import schedule
import datetime, time
import pandas as pd

import mlflow
from sklearn.preprocessing import normalize

import mysql.connector

def loadModel():
  mlflow.set_tracking_uri("http://localhost:5000")
  model_name      = "keras-ann-reg"
  stage           = "Production"
  loaded_model    = mlflow.pyfunc.load_model(f"models:/{model_name}/{stage}")
  return loaded_model

def getData(conn):
  dataIn  = pd.read_sql('SELECT * FROM iris', con=conn).sample(n=3)
  return dataIn

def getPrediction(conn):
  feature         = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
  dataIn          = getData(conn)
  dataIn[feature] = normalize(dataIn[feature], axis=0)
  model           = loadModel()
  datenow         = pd.Series([datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")]*len(dataIn.index))
  prediction      = model.predict(dataIn).to_numpy().argmax(axis=1)

  dataOut                 = pd.DataFrame(columns=['executed_at', 'id', 'class'])
  dataOut["executed_at"]  = pd.to_datetime(dataOut.executed_at.append(datenow, ignore_index=True))
  dataOut["id"]           = dataIn.id.values
  dataOut["class"]        = prediction
  return dataOut

def insertToTable():
  connection  = mysql.connector.connect(host='localhost', database='irisDB', user='root', password='admin123')
  prediction  = getPrediction(connection)
  
  if connection.is_connected():
    cursor  = connection.cursor()
    cursor.execute("select database();")
    record  = cursor.fetchone()

    for i, row in prediction.iterrows():
      cursor.execute("INSERT INTO irisDB.clf VALUES (%s,%s,%s)", tuple(row))
      connection.commit()

  else:
    connection.rollback()

  connection.close()

if __name__ == '__main__':
    #schedule.every(10).minutes.do(insertToTable)
    schedule.every().day.at("10:00").do(insertToTable)
    while True:
        schedule.run_pending()
        time.sleep(1)