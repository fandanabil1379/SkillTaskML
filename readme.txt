Steps:

1. Activate mysql server 

2. Set remote storage & artifact root (*adjust path and URL variables):

   mlflow server --backend-store-uri mysql+pymysql://root:admin123@localhost/mlflowDB --default-artifact-root file:/./Training/SkillTaskML/mlruns -h 0.0.0.0 -p 5000

3. Run the jupyter notebook
