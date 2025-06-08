import gradio as gr
import joblib

model = joblib.load("emotion_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")


def predict_emotion(text):
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)[0]
    return prediction


iface = gr.Interface(
    fn=predict_emotion,
    inputs=gr.Textbox(lines=2, placeholder="Enter a sentence expressing emotion..."),
    outputs="text",
    title="Emotion Classifier",
    description="Enter a sentence and the model will predict the emotion (anger, disgust, fear, joy, sadness, surprise)."
)

iface.launch()
