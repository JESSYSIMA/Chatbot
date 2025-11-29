from rest_framework.decorators import api_view
from rest_framework.response import Response
from transformers import pipeline


summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-6-6")
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr")

@api_view(['POST'])
def chat(request):
    text = request.data.get("text", "").strip()
    mode = request.data.get("mode", "").strip()  # "translate" ou "summarize"

    if not text:
        return Response({"response": "Veuillez fournir du texte."})

    if mode == "translate":
        reply = translator(text)[0]['translation_text']
    elif mode == "summarize":
        if len(text.split()) < 10:
            reply = " ".join(text.split()[:10]) + " ..."
        else:
            words = len(text.split())
            max_len = min(140, words)
            min_len = min(70, max_len)
            reply = summarizer(text, min_length=min_len, max_length=max_len, do_sample=False)[0]['summary_text']
    else:
        reply = "Mode inconnu"

    return Response({"response": reply})
