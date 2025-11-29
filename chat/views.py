from rest_framework.decorators import api_view
from rest_framework.response import Response
from transformers import pipeline
import gc

@api_view(['POST'])
def chat(request):
    text = request.data.get("text", "").strip()
    mode = request.data.get("mode", "").strip()  # "translate" ou "summarize"

    if not text:
        return Response({"response": "Veuillez fournir du texte."})

    reply = "Mode inconnu"

    if mode == "translate":
        # Charger la traduction seulement quand on en a besoin
        translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr")
        reply = translator(text)[0]['translation_text']
        del translator
        gc.collect()  # libère la mémoire

    elif mode == "summarize":
        if len(text.split()) < 10:
            reply = " ".join(text.split()[:10]) + " ..."
        else:
            words = len(text.split())
            max_len = min(140, words)
            min_len = min(70, max_len)
            # Charger le summarizer à la demande
            summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-6-6")
            reply = summarizer(text, min_length=min_len, max_length=max_len, do_sample=False)[0]['summary_text']
            del summarizer
            gc.collect()  # libère la mémoire

    return Response({"response": reply})
