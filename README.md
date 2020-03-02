# Disaster Response Pipeline Project

### Summary
this is a project from udacity about disaters. It consists of three steps:
1.we need create an ETL on data from two csv files and save the data results to a SQLite database. 
2.create a pipeline : bulid machine learning model, evaluate it and save it as a pickle file. 
3.display some data analysis of the disaters message and classify new messages from a search box.

### File Descriptions
app/run.py - start web server to run the web application
app/templates/go.html - Web page called when a search is executed
app/templates/master.html - Home web page; also performs some data analysis
data/DisasterResponse.db - SQLite database output by process_data.py after ETL process
data/disaster_categories.csv - Training data of categories
data/disaster_messages.csv - Training data of messages
data/process_data.py - Python code,read data and ETL
models/classifier.pkl - a model Pickle file created by run train_classifier.py
models/train_classifier.py - Trains and tests classification model,and save model as a file

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

