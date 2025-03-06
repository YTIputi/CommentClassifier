import torch
from models.CommentClassifier import CommentClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np


def main(com):
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(pd.read_csv('data/processed/train/train.csv')['Comment']).toarray()

    input_dim = X_train.shape[1]
    hidden_dim = 64
    output_dim = 3
    
    model = CommentClassifier(input_dim, hidden_dim, output_dim, num_lstm_layers=1)
    model.load_state_dict(torch.load('models/trained_comment_classifier.pth'))

    model.eval()

    x_com = np.array([com])
    x_vect = torch.tensor(vectorizer.transform(x_com).toarray(), dtype=torch.float32)

    outputs = model(x_vect)
    predicted = torch.argmax(outputs)

    if predicted == 0:
        print('negative')
    elif predicted == 1:
        print('neutral')
    else:
        print('positive')
    print(f'Предсказание модели: {outputs}')


if __name__ == "__main__":
    main('I hate these course')