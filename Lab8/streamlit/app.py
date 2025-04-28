import streamlit as st
import joblib
import pandas as pd

st.title("Reddit Comment Classification")
st.markdown("### All you have to do to use this app is enter a comment and hit the Predict button.")

reddit_comment = [st.text_area("Input your comment here:")]

def load_artifacts():
    model_pipeline = joblib.load("../reddit/reddit_model_pipeline.joblib")
    return model_pipeline

model_pipeline = load_artifacts()

def predict(reddit_comment):
    X = reddit_comment
    predictions = model_pipeline.predict(X)
    return predictions 

preds = predict(reddit_comment)
st.metric("Should this comment be removed (0: No; 1: Yes)", preds.round(2))

st.header("Get a Batch of Predictions")

batches = st.file_uploader("Upload File", type='csv')

if batches is not None:
    dataframe = pd.read_csv(batches, header=None).to_numpy().reshape((-1,))
    batch_predictions = pd.DataFrame(predict(dataframe))
    batch_predictions["Comment"] = dataframe
    batch_predictions.rename(columns={0:"Keep", 1:"Remove"}, inplace=True)
    st.write(batch_predictions)
    st.download_button('Download Predictions', data=batch_predictions.to_csv().encode('utf-8'), file_name='predictions.csv', mime='text/csv',)