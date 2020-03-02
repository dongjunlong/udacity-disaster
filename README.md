# Disaster Response Pipeline Project

### Summary
this is a project from udacity about disaters. It consists of three steps:
1.we need create an ETL on data from two csv files and save the data results to a SQLite database. 
2.create a pipeline : bulid machine learning model, evaluate it and save it as a pickle file. 
3.display some data analysis of the disaters message and classify new messages from a search box.

### Instructions:
How to work
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
        
       after this ,you can find DisasterResponse.db in data file
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
        
        after this,you can find classifier.pkl in models file

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

