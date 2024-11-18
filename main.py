import nltk
import sqlite3
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import RSLPStemmer
import spacy
from googletrans import Translator
import pandas as pd
from flask import Flask, request, jsonify, render_template
import random
import json

nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('vader_lexicon')

nlp = spacy.load('pt_core_news_sm')
main = Flask(__name__)

#Função de pré-processamento
def preprocessamento(texto):
    
    #Normalização do texto
    texto_normalizado = texto.lower()
    
    #Tokenização
    tokens = word_tokenize(texto_normalizado)

    #Remover stopwords
    stop_words = set(stopwords.words('portuguese'))
    tokens_sem_stopwords = [word for word in tokens if word not in stop_words]
    
    #Stemming
    stemmer = RSLPStemmer()
    tokens_stemmed = [stemmer.stem(word) for word in tokens_sem_stopwords]
    
    #Lematização
    lematizacao = [token.lemma_ for token in nlp(" ".join(tokens_sem_stopwords)).doc]
    
    detalhes = {
        "normalizacao": texto_normalizado,
        "tokens": tokens,
        "sem_stopwords": tokens_sem_stopwords,
        "stemming": tokens_stemmed,
        "lematizacao": lematizacao,
    }

    return detalhes

#Função de análise sentimental
def analiseSentimental(texto):
    # Pre-processamento do texto
    preprocessamento(texto)
     
    translator = Translator() 
    sia = SentimentIntensityAnalyzer()
    texto_traduzido = translator.translate(texto, src='pt', dest='en').text
    sentimento = sia.polarity_scores(texto_traduzido)
    if sentimento['compound'] >= 0.05:
        return "Texto Positivo", "positive"
    elif sentimento['compound'] <= -0.05:
         return "Texto Negativo", "negative"
    else:
         return "Texto Neutro","neutral"

# Função para pegar um texto aleatório da base de dados
def aleatorizar_texto():
    conn = sqlite3.connect('base_de_dados.db')    
    cursor = conn.cursor()
    cursor.execute("SELECT Review FROM Avaliações ORDER BY RANDOM() LIMIT 1")
    result = cursor.fetchall()
    conn.close()
    # Verifica se há resultado e se o valor não está vazio
    if result and result[0]:
        texto = result[0][0]  # Acessa o texto da coluna 'Review'
        return texto  # Acessa o valor real na tupla
    else:
        return "Nenhum texto encontrado na base de dados."


@main.route('/', methods=['GET', 'POST'])
def index():
    sentimento = None
    cor = None
    texto_original = None
    display_textarea = False
    detalhes = None
    erro = False

    if request.method == 'POST':
        action = request.form.get('action')

        if action == 'aleatorizar':
            texto_original = aleatorizar_texto()
            display_textarea = True 
        elif action == 'escrever':
            texto_original = ""
            display_textarea = True
        elif 'text' in request.form:
            texto_original = request.form['text']
            if texto_original.strip() == "": 
                erro = True
            else:
                detalhes = preprocessamento(texto_original)
                sentimento, cor = analiseSentimental(texto_original)
                display_textarea = True
                erro = False
        elif action == 'voltar':
            texto_original = ""
            sentimento = ""
            cor = ""
            detalhes = None
            erro = False
            display_textarea = True   

    return render_template(
        'index.html',
        texto_original=texto_original,
        detalhes=detalhes,
        sentimento=sentimento,
        cor=cor,
        erro=erro,
        display_textarea=display_textarea
    )

if __name__ == "__main__":
    main.run(debug=True)