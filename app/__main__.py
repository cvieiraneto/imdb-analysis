'''Arquivo principal do projeto.'''
from .funcs.preprocess import *
from .funcs.utils import *
from .funcs.openai import *
from concurrent.futures import ThreadPoolExecutor

pd.set_option('display.max_columns', 15)

def process_row(row):
    row['5words'] = get_movie_information(row['processed_overview'])
    row['5carac'] = get_movie_information(row['processed_overview'], num_words=5, prompt=f"Defina 5 adjetivos em português e separadas por vírgula que melhor representam o seguinte filme: \n'{row['processed_overview']}'\nLembre-se de separar os adjetivos por vírgulas.")
    return row

@timer
def my_program():
    '''Função principal do projeto.'''
    df = pre_process(r'dados\tmdb_movies_dataset.csv')
    #normaliza as colunas vote_average, vote_count e popularity
    # df['vote_average'] = df['vote_average']-df['vote_average'].mean()/(df['vote_average'].max()-df['vote_average'].min())
    df['vote_count'] = df['vote_count']-df['vote_count'].mean()/(df['vote_count'].max()-df['vote_count'].min())
    df['popularity'] = df['popularity']-df['popularity'].mean()/(df['popularity'].max()-df['popularity'].min())
    df['score'] = 0.5*df['vote_average'] + 0.4*df['vote_count'] + (0.1*df['popularity'])
    df.sort_values(by='score', ascending=False, inplace=True)
    #select top 20
    df = df.iloc[0:100, :]
    df = preprocess_text('overview', df)
    # df['5words'] = df['processed_overview'].apply(lambda x: get_movie_genre(x))
    # df['5carac'] = df['processed_overview'].apply(lambda x: get_movie_genre(x, num_words=5, prompt=f"Defina 5 características em português separadas por vírgula que melhor representam o seguinte filme: '{x}'\nLembre-se de separar as características por vírgulas."))
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(process_row, df.to_dict('records')))
    df = pd.DataFrame(results)
    df.to_excel(r'dados\output.xlsx', index=False)
    return df

if __name__ == '__main__':
    # nltk.download('stopwords', download_dir=r'dados\nltk_data')
    df = my_program()
    # print('Fim do programa.')
    # Passando a coluna "5carac" como argumento
    gerar_nuvem_palavras(df['5carac'])

