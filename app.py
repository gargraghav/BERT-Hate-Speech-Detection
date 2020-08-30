import pandas as pd

import numpy as np

import torch

import transformers
from transformers import BertForSequenceClassification, BertConfig
from transformers import BertTokenizer

from keras.preprocessing.sequence import pad_sequences

#web frameworks
from starlette.applications import Starlette
from starlette.responses import JSONResponse, HTMLResponse, RedirectResponse
import uvicorn
import aiohttp
import asyncio
import os
import sys

MODEL_TYPE = "bert-base-multilingual-uncased"
CASE_BOOL = True # do_lower_case=CASE_BOOL
MAX_LEN = 256
app = Starlette()


def preprocess_for_bert(sentences, MAX_LEN):
    
    print('Started pre-processing...')

    
    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []

    # For every sentence...
    for sent in sentences:
        # `encode` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        encoded_sent = tokenizer.encode(
                            sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'

                            # This function also supports truncation and conversion
                            # to pytorch tensors, but we need to do padding, so we
                            # can't use these features :( .
                            #max_length = 128,          # Truncate all sentences.
                            #return_tensors = 'pt',     # Return pytorch tensors.
                       )

        # Add the encoded sentence to the list.
        input_ids.append(encoded_sent)
        
    
    padded_input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", 
                              value=0, truncating="post", padding="post")
    
    for sent in padded_input_ids: # go row by row through the numpy 2D array.
        length = len(sent)
        
        if (sent[length-1] != 0) and (sent[length-1] != 102): # 102 is the SEP token
            sent[length-1] = 102 # set the last value to be the SEP token i.e. 102
    
    
    # Create attention masks
    attention_masks = []

    # For each sentence...
    for sent in padded_input_ids:

        # Create the attention mask.
        #   - If a token ID is 0, then it's padding, set the mask to 0.
        #   - If a token ID is > 0, then it's a real token, set the mask to 1.
        att_mask = [int(token_id > 0) for token_id in sent]

        # Store the attention mask for this sentence.
        attention_masks.append(att_mask)
        
        
    print('Finished pre-processing...')
    return padded_input_ids, attention_masks

def make_prediction(sentence, model):
    
    
    # convert to a list
    sentence = [sentence]
    
    # pre-process
    padded_token_list, att_mask = preprocess_for_bert(sentence, MAX_LEN)
    
    print('Zero')
    
    # convert to torch tensors
    padded_token_list = torch.tensor(padded_token_list, dtype=torch.long)
    att_mask = torch.tensor(att_mask, dtype=torch.long)
    
    print('One')
    
    # make a prediction
    outputs = model(padded_token_list, 
                token_type_ids=None, 
                attention_mask=att_mask)
    
    print('Two')
    
    # get the preds
    preds = outputs[0]
    
    # convert to probabilities
    preds_proba = torch.sigmoid(preds)
    
    print('Three')
    
    # convert to numpy
    np_preds = preds_proba.detach().cpu().numpy()
    
    # get the first row
    np_preds = np_preds[0]
    
    print('Four')
    
    # extract the probailities for each class
    not_toxic_proba = np_preds[0]
    toxic_proba = np_preds[1]
    
    print('Finished prediction...')
    return not_toxic_proba, toxic_proba

device = 'cpu'


# Instantiate the tokenizer
# ..........................

tokenizer = BertTokenizer.from_pretrained(MODEL_TYPE, do_lower_case=CASE_BOOL)
print('Tokenizer loaded.')

# Load the Model
# ...............

print('Local model is initializing...')

model = BertForSequenceClassification.from_pretrained(MODEL_TYPE, num_labels = 2,
                                                output_attentions = False, 
                                                output_hidden_states = False)
	
print('Model initialization complete.')

print('Model is loading...')
# Load the saved weights into the architecture.
# Note that this file (views.py) gets imported into the app.ini file therefore,
# place the model in the same folder as the app.ini file.
path = 'model_final.pt'
model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

print('../Model loaded.')

# Send the model to the device.
model.to(device)

@app.route("/")

def form(request):
    return HTMLResponse(
            """
            <h1> Hate Speech Detection </h1>
            <br>
            <u> Submit Text </u>
            <form action = "/classify-text" method="get">
                1. <input type="text" name="text" size="60"><br><p>
                2. <input type="submit" value="Submit">
            </form>
            """) 


@app.route("/form")
def redirect_to_homepage(request):
        return RedirectResponse("/")

@app.route('/classify-text', methods = ["GET"])
def classify_text(request):
    message = request.query_params["text"]
    return predict(message)

def predict(message):
    sentence = message
    print(sentence)

    not_toxic_proba, toxic_proba = make_prediction(sentence, model)
    
    toxic_proba = toxic_proba * 100
	
    toxic_proba = np.round(toxic_proba, 2) 
	
	# Convert to type string because json cannot work with numpy float32.
    toxic_proba = str(toxic_proba)
    # print(toxic_proba)
    return HTMLResponse(
        """
        <html>
            <body>
                <p> Toxicity: <b> %s </b> </p>
            </body>
        </html>
        """ %(toxic_proba))



if __name__ == "__main__":
    if "serve" in sys.argv:
        port = int(os.environ.get("PORT", 8008)) 
        uvicorn.run(app, host = "0.0.0.0", port = port)