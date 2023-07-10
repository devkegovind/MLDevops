import mlflow

def calculate_multiple(x, y):
    return x*y



if __name__== '__main__':
    #starting the server of mlflow
    with mlflow.start_run():
        x,y = 10,20
        z = calculate_multiple(x,y)
        
        #Tracking the experiment with the mlflow
        mlflow.log_param("x",x)
        mlflow.log_param("y",y)
        mlflow.log_metric("z",z)   