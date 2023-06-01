'''Arquivo de pré-processamento de dados.'''
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer, SnowballStemmer
from langdetect import detect
import iso639
from multiprocessing import Pool, cpu_count
import unicodedata

nltk.data.path.append(r".\dados\nltk_data")

def pre_process(path):
    '''Função de pré-processamento de dados.'''
    df = pd.read_csv(path, sep=',')
    df = df.dropna()
    df = df.drop_duplicates()
    return df
 
def process_string(string):
    """
    Função que realiza pré-processamento em um texto.

    Args:
        string (str): O texto a ser pré-processado.
        idioma (str, optional): O idioma da stopword list e do stemmer. Padrão é 'portuguese'.

    Returns:
        str: O texto pré-processado.
    """
    # Converte para minúsculas
    string = string.lower()

    # Prepara o regex para remover pontuações
    punct_regex = re.compile('[^\w\s]')

    # Remove pontuações
    string = re.sub(punct_regex, '', string)

    #Remove acentos
    string = unicodedata.normalize('NFKD', string).encode('ASCII', 'ignore').decode('ASCII')

    #Prepara regex para remover números
    num_regex = re.compile('\d+')

    # Remove números
    string = re.sub(num_regex, '', string)

    # Detecta o idioma do texto
    idioma = (iso639.to_name(detect(string))).lower()

    # Define as stopwords de acordo com o idioma
    # Portugues
    if idioma == 'portuguese':
        stop_words = set(stopwords.words('portuguese'))
        stemmer = PorterStemmer()
    # Inglês
    elif idioma == 'english':
        stop_words = set(stopwords.words('english'))
        stemmer = SnowballStemmer('english')
    # Outros idiomas
    else:
        try:
            stop_words = set(stopwords.words(idioma))
            stemmer = SnowballStemmer(idioma)
        except:
            stop_words = set(stopwords.words('portuguese'))
            stemmer = PorterStemmer()
            # raise ValueError(f"Identificado idioma:{idioma}. Idioma deve ser 'portuguese' ou 'english'.")
    
    # Remove stopwords    
    string = ' '.join([word for word in string.split() if word not in stop_words])

    # # Aplica stemming
    # string = ' '.join([stemmer.stem(word) for word in string.split()])

    #Remove palavras com menos de 3 caracteres
    string = ' '.join([word for word in string.split() if len(word) > 3])
    
    return string


def preprocess_text(col_name, df):
    """
    Pré-processa o texto presente na coluna especificada de um dataframe.
    
    Args:
        col_name (str): Nome da coluna a ser pré-processada.
        df (pandas.DataFrame): Dataframe contendo a coluna a ser pré-processada.
        
    Returns:
        pandas.DataFrame: Dataframe com uma nova coluna "processed_{col_name}" contendo o texto pré-processado.
    """
    
    # Dividir o dataframe em partes iguais para cada processo
    num_processes = cpu_count()
    print(f"Dividindo o dataframe em {num_processes} partes iguais para processamento paralelo.")
    chunk_size = len(df) // num_processes
    if chunk_size == 0:
        chunk_size = 1
    chunks = [df[i:i+chunk_size] for i in range(0, len(df), chunk_size)]

    # Iniciar os processos
    with Pool(num_processes) as pool:
        # Aplicar a função de pré-processamento para cada parte do dataframe
        results = pool.starmap(process_text_chunk, [(chunk, col_name) for chunk in chunks])
    
    # Juntar os resultados em um único dataframe
    df = pd.concat(results)

    return df

# Função que será aplicada a cada parte do dataframe
def process_text_chunk(chunk, col_name):
    chunk[f'processed_{col_name}'] = chunk[col_name].apply(lambda text: process_string(text))
    # chunk[f'embedding_{col_name}'] = chunk[f'processed_{col_name}'].apply(lambda text: translate_and_embed(text))
    return chunk