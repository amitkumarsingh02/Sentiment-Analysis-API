class app_config:
    data_path = 'airline_sentiment_analysis.csv'
    token_path = 'token.pickle'
    model_path = "BiLSTM_Model"
    random_state = 42
    max_pad_len = 25


    # Training Parameters
    validation_split= 0.0
    epochs= 25
    batch_size= 32
    
    # Model Parameters
    embedding_dim = 5000
    output_dim = 32
    learning_rate= 0.01
    regularizers = 0.001
    input_len = 23