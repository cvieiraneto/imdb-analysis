import openai
import os
from .utils import timer

# Definir credenciais da API do OpenAI GPT
openai.api_key = os.environ['OPENAI_API_KEY']

def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

def get_translation(text, source_lang="en", target_lang="pt"):
    return openai.Completion.create(
        model="text-davinci-003",
        prompt=f"Traduza o texto abaixo do {source_lang} para o {target_lang}:\n{text}",
        temperature=0.3,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )

def get_movie_information(text, num_words=5, prompt=None):
    if prompt == None:
        prompt = f"Defina {num_words} palavras em português separadas por vírgula que melhor representam o conteúdo deste filme: '{text}'\nLembre-se de separar as palavras com vírgulas."
        # prompt = f"Defina as {num_words} palavras que melhor representam o conteúdo desse filme:" + text + "\n"
    completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    max_tokens=30,
    temperature=0.2,
    messages=[
        {"role": "user", "content": prompt}
    ]
    )
    return completion.choices[0].message['content']
