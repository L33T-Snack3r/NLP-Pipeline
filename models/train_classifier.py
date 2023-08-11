import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
# import xgboost as xgb
# from hyperopt import fmin, tpe, hp, STATUS_OK
import timeit
import pickle

def load_data(database_filepath):
    """
    This function loads in the cleaned messages and categories stored in the database file

    Input:
        database_filepath : filepath to location where database is stored
    Output
        X: Dataframe containing just the messages
        Y: Dataframe containing the column categories
        Y.columns: category labels
    """

    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('MessageCategories',con=engine)
    
    #X and Y
    X = df['message'] 
    Y = df[df.columns.difference(['id','message','genre','original'])]
    
    return X, Y, Y.columns
    pass


def tokenize(text):
    """
    Custom tokenization function. This function:
    1. replaces urls with a custom placeholder
    2. Lemmatizes
    3. reduces to lower case
    4. removes spaces
    5. tokenizes (converts sentences to individual words)

    Input:
        text : string containing a message
    Output
        clean_tokens: list containing cleaned tokens
    """
    
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens
    pass

####Attempt to inplement XGBoost

# def optimize_hyperpars(X, Y):
#     X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = .25, random_state=666)
    
#     # Define the objective function to minimize
#     def objective(params):
#         xgb_model = xgb.XGBClassifier(**params)
#         pipeline = Pipeline([
#                         ('vect', CountVectorizer(tokenizer=tokenize)),
#                         ('tfidf', TfidfTransformer()),
#                         ('clf', MultiOutputClassifier(xgb_model))
#                         ])
#         pipeline.fit(X_train, y_train)
#         y_pred = pipeline.predict(X_test)

#         #The loss function here is the one I defined previously
#         score = custom_loss(y_test,y_pred)

#         #The score is returned here as a negative value, as fmin will attempt to minimize this value
#         return {'loss': -score, 'status': STATUS_OK}
    
#     space = {
#         'max_depth': hp.choice('max_depth', range(5, 30, 1)),
#         'learning_rate': hp.loguniform('learning_rate', -5, -2),
#         'subsample': hp.uniform('subsample', 0.5, 1),
#         'n_estimators' : hp.choice('n_estimators', range(5, 50, 1)),
#         'reg_lambda' : hp.uniform ('reg_lambda', 0,1),
#         'reg_alpha' : hp.uniform ('reg_alpha', 0,1)
#     }
    
#     best_params = fmin(objective, space, algo=tpe.suggest, max_evals=1)
#     print("Best set of hyperparameters: ", best_params)
    
#     return best_params
#     pass

# def build_model(X,Y):
#     best_params = optimize_hyperpars(X,Y)
#     classifier = xgb.XGBClassifier(learning_rate=best_params['learning_rate'],
#                                            max_depth=best_params['max_depth'],
#                                            n_estimators=best_params['n_estimators'],
#                                            reg_alpha=best_params['reg_alpha'],
#                                            reg_lambda=best_params['reg_lambda'],
#                                            subsample=best_params['subsample'])
#     pipeline = Pipeline([
#                         ('vect', CountVectorizer(tokenizer=tokenize)),
#                         ('tfidf', TfidfTransformer()),
#                         ('clf', MultiOutputClassifier(classifier))
#                         ])
#     return pipeline

# def custom_loss(y_test,y_pred):
#     score = 0
    
#     for i,col in enumerate(y_test.columns):
#         y_pred_col = y_pred[:,i]
#         y_test_col = y_test[col].values
#         x = f1_score(y_test_col, y_pred_col, average='macro')
#         score = score + (x)**2

#     score = np.sqrt(score/y_test.shape[1])
#     return score

#######

def build_model():
    """
    This function:
    1. Builds a pipeline consisting of a count vectorizer, tfidf transformer and 
        multioutputclassifier using a random forest classifier
    2. Uses gridsearch to determine the best parameters from a small subset of possible options.
        This subset of potential parameters is kept small to allow the program to complete in ~15 minutes

    Input:
        None
    Output
        cv : pipeline with the best parameters
    """
    
    classifier = RandomForestClassifier()
    pipeline = Pipeline([
        ('vect',CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf',MultiOutputClassifier(classifier))
    ])
    
    parameters = {
        'vect__ngram_range': ((1,1),(1,2)),
        'clf__estimator__n_estimators':[5,10,20]
    }
    
    cv = GridSearchCV(pipeline, param_grid=parameters, cv=2, n_jobs=6)
    
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """
    This function:
    Evaluates the model by printing out the classification report (precision, recall, f1 score)
    for each category

    Input:
        model : the best model determined through gridsearch
        X_test : test set of messages
        Y_test : categories corresponding to the test set messages
        category_names : the category labels
    Output
        None
    """

    Y_pred = model.predict(X_test)
    for i,col in enumerate(category_names):
        Y_pred_col = Y_pred[:,i]
        Y_test_col = Y_test[col].values
        print('col: ' + col)
        print(classification_report(Y_test_col, Y_pred_col))
    pass


def save_model(model, model_filepath):
    """
    This function saves the model to a pickle file

    Input:
        model : the machine learning model
        model_filepath: filepath to location where model will be saved
    Output
        None
    """

    pickle.dump(model, open(model_filepath, 'wb'))
    pass


def main():
    if len(sys.argv) == 3:
        st1 = timeit.default_timer()
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!\n')
        
        st2 = timeit.default_timer()
        print('The run took {} s'.format(st2-st1))

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()