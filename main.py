from training import Training
from preprocessing import Preprocessing
from app_config import app_config
from sklearn.model_selection import train_test_split
import pickle

def main():
    print("======================= Cleaning Data Started ================================")
    preprocessing = Preprocessing()
    data = preprocessing.load_data(app_config)
    data['airline_sentiment'] = data['airline_sentiment'].apply(preprocessing.label_to_index)
    data['text'] = data['text'].apply(preprocessing.clean_text)
    data['text'] = data['text'].apply(preprocessing.lemmatizing)
    print("======================= Cleaning Data Completed ================================")

    print("======================= Training Data Started ================================")
    training = Training()
    token = training.tokenize_text(data['text'])
    data['tokenized_text'] = training.padded_tokenized_text
    model = training.build_bilstm_model(app_config)
    print(model.summary())
    history = training.train_model(model, data['tokenized_text'], data['airline_sentiment'], epochs=app_config.epochs, batch_size=app_config.batch_size)
    pickle.dump(token, open(app_config.token_path, 'wb'))
    model.save(app_config.model_path)
    print("======================= Training Completed ================================")

if __name__ == '__main__':
    main()