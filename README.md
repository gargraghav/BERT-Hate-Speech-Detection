# BERT-Hate-Speech-Detection

This project aims to build and deploy a production-grade Hate speech detection model. Users are able to send text to the model, via an API, and get back predictions.

I used a powerful language model like Bert and deployed it as a microservice. Websites can send text to a cloud server (hosted on GCP). The server responds by sending back a hate speech prediction for each piece of text. The system can also be set up to allow a user to send a file containing rows of text. The model will process this file and the server will return a new file with a prediction for each row.

The model AUC score is **0.85**.
