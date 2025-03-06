import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def normalize_text(data):
    n_min = np.min(filtered_data.groupby('Sentiment').count())

    data_pos = data.loc[np.random.choice(data[data['Sentiment'] == 'positive'].index, n_min), :].reset_index().drop('index', axis=1)
    data_neu = data.loc[np.random.choice(data[data['Sentiment'] == 'neutral'].index, n_min), :].reset_index().drop('index', axis=1)
    data_neg = data.loc[np.random.choice(data[data['Sentiment'] == 'negative'].index, n_min), :].reset_index().drop('index', axis=1)

    new_data = pd.concat([data_pos, data_neu, data_neg], axis=0).reset_index().drop('index', axis=1)
    data_dummies = pd.get_dummies(new_data, columns=['Sentiment'])
    return data_dummies


def split_data(data, p):
    X_train, X_test, y_train, y_test = train_test_split(data['Comment'].to_numpy(), data.drop('Comment', axis=1).to_numpy(), train_size=p)
    return X_train, X_test, y_train, y_test


def filter_english_comments(data):
    return data[data['Comment'].str.contains(r'^[a-zA-Z0-9\s]+$', regex=True)]

def remove_missing_values(data):
    return data.dropna()

def save_data(X_train, X_test, y_train, y_test):
    train_df = pd.DataFrame({
        'Comment': X_train,
        'Sentiment_negative': y_train[:, 0],
        'Sentiment_neutral': y_train[:, 1],
        'Sentiment_positive': y_train[:, 2]
    })
    
    test_df = pd.DataFrame({
        'Comment': X_test,
        'Sentiment_negative': y_test[:, 0],
        'Sentiment_neutral': y_test[:, 1],
        'Sentiment_positive': y_test[:, 2]
    })
    
    train_df.to_csv('data/processed/train/train.csv', index=False)
    test_df.to_csv('data/processed/test/test.csv', index=False)


if __name__ == "__main__":
    data = pd.read_csv('data/raw/YoutubeCommentsDataSet.csv')
    remove_data = remove_missing_values(data)
    filtered_data = filter_english_comments(remove_data)
    cleaned_data = normalize_text(filtered_data)
    X_train, X_test, y_train, y_test = split_data(cleaned_data, .8)
    save_data(X_train, X_test, y_train, y_test)
