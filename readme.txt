Steps:
1. pip install -r requirements.txt

2. Activate mysql server 

3. Set remote storage & artifact root (*adjust path and URL variables):

   mlflow server --backend-store-uri mysql+pymysql://root:admin123@localhost/mlflowDB --default-artifact-root file:/./SkillTaskML/mlruns -h 0.0.0.0 -p 5000

4. Run the jupyter notebook
