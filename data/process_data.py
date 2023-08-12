import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    This function loads and merges the messages and categories datasets on message id.

    Input:
        messages_filepath : filepath to the csv file containing disaster messages
        categories_filepath : filepath to the csv file containing the categories of the disaster messages
    Output:
        df : Dataframe consisting of merged messages and categories
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    #merge messages and categories dataframes on id key
    df = messages.merge(categories, how='inner', on=['id'])
    return df
    pass


def clean_data(df):
    """
    This function cleans the merged dataframe by:
    1. setting appropriate column labels
    2. extracting numeric data e.g. related - 1 ---> 1
    3. converting from string to numeric
    4. convert all 2s to 1s

    Input:
        df : Merged dataframe of messages and categories
    Output:
        df : Cleaned dataframe
    """

    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat=';',expand=True)
    
    # select the first row of the categories dataframe
    row = pd.DataFrame(categories.iloc[0,:])
    
    # use this row to extract a list of new column names for categories.
    category_colnames = row[0].apply(lambda x : x[:-2])
    
    #rename categories column names to cleaned column names
    categories.columns = category_colnames
    
    #convert category values to just 1 and 0s
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x : x[-1])
    
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

        # Some values have 0, 1 and 2. This converts all values which are 2s into 1s
        categories.loc[(categories[column] == 2),column] = 1

    # drop the original categories column from `df`
    df = df.drop(columns=['categories'])
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)
    
    # drop duplicates
    df = df.drop_duplicates(subset=list(df.columns.difference(['id'])))
    
    return df
    pass


def save_data(df, database_filename):
    """
    This function saves the cleaned dataframe to a .db file

    Input:
        df : cleaned dataframe
        database_filename: filepath where database file will be saved
    Output:
        None
    """

    out_path = 'sqlite:///' +  database_filename
    engine = create_engine(out_path)
    df.to_sql('MessageCategories', engine, index=False, if_exists='replace')
    pass  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()