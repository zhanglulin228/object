from flask import Flask, request, jsonify
import pickle
from sentence_transformers import SentenceTransformer
app = Flask(__name__)
# Load the saved model
with open('url_check_model.pkl', 'rb') as f:
    model = pickle.load(f)
# Load the SentenceTransformer model
SentenceTransformer_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
# Define the URL check endpoint
@app.route('/api/url_check', methods=['POST'])
def url_check(url):
    # Get the URL from the request
    url = request.json['url']

    # Embed the URL using the SentenceTransformer model
    embedding = SentenceTransformer_model.encode(url)
    embedding = embedding.reshape(1, -1)

    # Predict the URL using the loaded model
    prediction = model.predict(embedding)

    # Map the prediction to the output format
    output = {'result': 'good' if prediction[0] == 1 else 'bad'}
    
    #print("Prediction for new URL: ", output)

    # Return the output as a JSON response
    return jsonify(output)
if __name__ == '__main__':
    app.run()