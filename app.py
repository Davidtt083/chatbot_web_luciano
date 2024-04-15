from dotenv import load_dotenv
load_dotenv() ## loading all the environment variables
from flask import Flask, render_template, request, session, jsonify
import google.generativeai as genai
from flow.responses import get_gemini_response 
import os
from gemini.promts import instruccion2,documents
import google.generativeai as genai
from gtts import gTTS
import tempfile
import pandas as pd
import numpy as np
from IPython.display import display
from IPython.display import Markdown
import textwrap
import markdown
from markupsafe import Markup
import re

safety_settings = [
    {
        "category": "HARM_CATEGORY_DANGEROUS",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]

model = genai.GenerativeModel('gemini-pro')
model_embedding = 'models/embedding-001'
chat = model.start_chat(history=[])
conversations = []
instruction = instruccion2
AUDIO_FOLDER = 'templates/audio_files'
if not os.path.exists(AUDIO_FOLDER):
    os.makedirs(AUDIO_FOLDER)
    
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

def clean_text(text):
    """
    Función para limpiar el texto y eliminar las etiquetas HTML.
    """
    # Eliminar las etiquetas HTML utilizando expresiones regulares
    cleaned_text = re.sub(r'<[^>]+>', '', text)
    return cleaned_text

app = Flask(__name__, static_folder='templates')
conversations = []



@app.route('/', methods=['GET', 'POST'])
def home():
  
    if request.method == 'GET':
        return render_template('index.html', conversations=conversations)

    elif request.method == 'POST':
        
        df = pd.DataFrame(documents)
        df.columns = ['Title', 'Text']
        
        
        def embed_fn(title, text):
            return genai.embed_content(model=model_embedding,
                             content=text,
                             task_type="retrieval_document",
                             title=title)["embedding"]

        df['Embeddings'] = df.apply(lambda row: embed_fn(row['Title'], row['Text']), axis=1)
               
        query = request.form['question']
       
        request_e = genai.embed_content(model=model_embedding,
                              content=query,
                              task_type="retrieval_query")
        
        def find_best_passage(query, dataframe):
            """
            Compute the distances between the query and each document in the dataframe
            using the dot product.
            """
            query_embedding = genai.embed_content(model=model_embedding,
                                        content=query,
                                        task_type="retrieval_query")
            dot_products = np.dot(np.stack(dataframe['Embeddings']), query_embedding["embedding"])
            idx = np.argmax(dot_products)
            return dataframe.iloc[idx]['Text'] # Return text from index with max value
        
        passage = find_best_passage(query, df)
        print(passage)
        
        question = request.form['question']
        response = chat.send_message(question + "\n\nResponde únicamente con la siguiente información: " + passage + "\n\n" + instruction, generation_config=genai.types.GenerationConfig(temperature=0.2), safety_settings=safety_settings)
        print(response.text)
        
        bot_response = response.text
                
        response_lines = [Markup(line) for line in bot_response.split('\n') if line.strip()]
        conversations.append({'user': question, 'bot': response_lines})
        
        audio_filename = f"{len(conversations)}.mp3"
        audio_path = os.path.join(AUDIO_FOLDER, audio_filename)
        
        # Limpiar el texto antes de generar el audio
        cleaned_bot_response = clean_text(bot_response)
        tts = gTTS(text=cleaned_bot_response, lang='es')
        tts.save(audio_path)
        
        return jsonify({'response': response_lines, 'audio_file': audio_path})
    

    
if __name__ == '__main__':
    app.run(debug=True, port=int(os.environ.get('PORT', 4000)))