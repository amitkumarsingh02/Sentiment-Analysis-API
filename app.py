import pickle
import uvicorn
from fastapi import FastAPI, Path
from app_config import app_config
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = FastAPI()
model = load_model(app_config.model_path)
token = pickle.load(open(app_config.token_path, 'rb'))

class HomeAPI:

    @app.get("/")
    def home():
        return {"Home":"Sentiment Analysis API"}

    @app.get("/check/{text}")
    def check_sentiment( text:str = Path(None, description="Need text to pedict Sentiment")):
        tokenized_text = token.texts_to_sequences([text])
        padded_tokenized_text = pad_sequences(tokenized_text, app_config.input_len).tolist()
        prob = model.predict(padded_tokenized_text)[0][0]
        sentiment = 'positive' if prob >= 0.5 else 'negative'
        response = {'sentiment': sentiment , 'probability' : prob }
        return str(response)

# Run the app
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=9050)