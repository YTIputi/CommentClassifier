import pandas as pd
from models.CommentClassifier import CommentClassifier
from models.CommentDataset import CommentDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer


def to_vector(X_train, y_train, X_test, y_test, vectorizer):
    X_train = vectorizer.fit_transform(X_train).toarray()
    X_test = vectorizer.transform(X_test).toarray()
    y_train, y_test = y_train.to_numpy(), y_test.to_numpy()
    return X_train, y_train, X_test, y_test

def load_data(train_path, test_path):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    X_train, y_train = train_data['Comment'], train_data.drop('Comment', axis=1)
    X_test, y_test = test_data['Comment'], test_data.drop('Comment', axis=1)
    
    return X_train, y_train, X_test, y_test

def prepare_datasets(X_train, y_train, X_test, y_test):
    dataset_train = CommentDataset(X_train, y_train)
    dataset_val = CommentDataset(X_test, y_test)
    
    return dataset_train, dataset_val

def create_data_loaders(dataset_train, dataset_val, batch_size):
    data_loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    data_loader_val = DataLoader(dataset_val, batch_size=batch_size)
    
    return data_loader_train, data_loader_val

def train_model(model, device, data_loader_train, criterion, optimizer, epochs):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in data_loader_train:
            text = batch['text'].to(device)
            label = batch['label'].to(device)

            optimizer.zero_grad()

            outputs = model(text)
            loss = criterion(outputs, label)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f'Эпоха {epoch+1}, потеря: {total_loss / len(data_loader_train)}')

def evaluate_model(model, data_loader_val):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in data_loader_val:
            input_ids = batch['text']
            labels = batch['label']
            outputs = model(input_ids)
            _, predicted = torch.max(outputs, dim=1)
            _, actual = torch.max(labels, dim=1)


            correct += (predicted == actual).sum().item()
            total += len(labels)

    accuracy = correct / total
    print(f'Точность на валидационном наборе: {accuracy}')

def save_model(model, path):
    torch.save(model.state_dict(), path)


def main():
    train_path = 'data/processed/train/train.csv'
    test_path = 'data/processed/test/test.csv'
    
    X_train, y_train, X_test, y_test = load_data(train_path, test_path)
    
    vectorizer = TfidfVectorizer()
    X_train, y_train, X_test, y_test = to_vector(X_train, y_train, X_test, y_test, vectorizer)

    batch_size = 32
    
    dataset_train, dataset_val = prepare_datasets(X_train, y_train, X_test, y_test)
    data_loader_train, data_loader_val = create_data_loaders(dataset_train, dataset_val, batch_size)

    input_dim = X_train.shape[1]
    hidden_dim = 64
    output_dim = 3
    model = CommentClassifier(input_dim, hidden_dim, output_dim, num_lstm_layers=1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 150
    train_model(model, device, data_loader_train, criterion, optimizer, epochs)
    
    evaluate_model(model, data_loader_val)
    
    save_path = 'models/trained_comment_classifier.pth'
    save_model(model, save_path)
    print(f'Модель сохранена в {save_path}')

if __name__ == "__main__":
    main()
