### Funções de utilização geral
import time
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from .preprocess import process_string

def timer(func):
    """
    Decorator que mede o tempo de execução de uma função.

    Parâmetros:
        func (callable): Função que será decorada.

    Retorna:
        callable: A função original com uma medida adicional de tempo de execução.

    """
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Tempo de execução de {func.__name__}: {end - start} segundos")
        return result
    return wrapper


def gerar_nuvem_palavras(texto):
    """
    Essa função recebe um texto e gera uma nuvem de palavras com as 10 mais frequentes 
    e também plota um histograma dessas palavras em relação à frequência.

    Args:
        texto (str): Texto a ser utilizado para gerar a nuvem de palavras.

    Returns:
        None. A função apenas exibe os gráficos.
    """
    # Transforma o texto em uma única string
    texto_string = ' '.join(texto)

    # Preprocessa o texto
    texto_string = process_string(texto_string)
    
    # Cria a nuvem de palavras
    cloud = WordCloud(background_color='white', width=1600, height=800).generate(texto_string)

    # Dividir o texto em palavras
    palavras = texto_string.split()
    
    # Contar a frequência de cada palavra
    contagem = Counter(palavras)
    
    # Obter as 5 palavras mais frequentes
    palavras_mais_frequentes = contagem.most_common(10)
    
    # Extrair as palavras e as contagens
    palavras, frequencias = zip(*palavras_mais_frequentes)

    plt.figure(figsize=(12, 6))  # Aumenta o tamanho da imagem
    
    # Plota o histograma
    plt.subplot(121)
    plt.bar(palavras, frequencias)
    plt.title('Top 10 Palavras Mais Frequentes')
    plt.xlabel('Palavras')
    plt.ylabel('Frequência')
    plt.xticks(rotation=45, ha='right')  # Rotaciona os textos do eixo x

    # Plota a nuvem de palavras
    plt.subplot(122)
    plt.imshow(cloud, interpolation="bilinear")
    plt.axis("off")

    
    plt.show()
