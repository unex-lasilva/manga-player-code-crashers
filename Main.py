import pandas as pd
import numpy as np
import os


# Carregamento e Pré-processamento de Dados
def carregar_dados(caminho="./archive"):
    """Carrega todos os arquivos Excel e os retorna como DataFrames"""
    print("Carregando dados...")

    # Carrega avaliações
    avaliacoes = pd.read_csv(os.path.join(caminho, "ratings_small.csv"))

    # Carrega metadados dos filmes
    filmes = pd.read_csv(os.path.join(caminho, "movies_metadata.csv"),
                         dtype={"popularity": str},
                         low_memory=False)

    # Carrega links
    links = pd.read_csv(os.path.join(caminho, "links_small.csv"))

    print("Dados carregados com sucesso!")
    return avaliacoes, filmes, links


def pre_processar_dados(caminho="./archive"):
    """
    Realiza o pré-processamento dos dados, criando um dataset unificado
    com apenas filmes que possuem avaliações maiores que 3
    """
    # Carregando dados
    avaliacoes, filmes, links = carregar_dados(caminho)

    print(f"Formato original dos dados:")
    print(f"Avaliações: {avaliacoes.shape} linhas x colunas")
    print(f"Filmes: {filmes.shape} linhas x colunas")
    print(f"Links: {links.shape} linhas x colunas")

    avaliacoes_filtradas = avaliacoes[avaliacoes['rating'] > 3]

    filmes_com_boas_avaliacoes = avaliacoes_filtradas['movieId'].unique()
    print(f"Encontrados {len(filmes_com_boas_avaliacoes)} filmes com pelo menos uma avaliação > 3")

    filmes = filmes[pd.to_numeric(filmes['id'], errors='coerce').notna()]
    filmes['id'] = filmes['id'].astype(int)
    links = links[pd.to_numeric(links['movieId'], errors='coerce').notna()]
    links['movieId'] = links['movieId'].astype(int)
    filmes_com_links = pd.merge(
        filmes,
        links,
        left_on='id',
        right_on='movieId',
        how='inner'
    )
    print(f"\nFilmes com links estabelecidos: {filmes_com_links.shape[0]}")

    filmes_com_links_filtrados = filmes_com_links[
        filmes_com_links['movieId'].isin(filmes_com_boas_avaliacoes)
    ]
    print(f"Filmes com links e boas avaliações: {filmes_com_links_filtrados.shape[0]}")

    avaliacoes_dataset_final = avaliacoes_filtradas[
        avaliacoes_filtradas['movieId'].isin(filmes_com_links_filtrados['movieId'])
    ]

    colunas_importantes = ['id', 'title', 'original_title', 'release_date',
                           'genres', 'overview', 'vote_average', 'movieId']

    filmes_info_compacta = filmes_com_links_filtrados[
        [col for col in colunas_importantes if col in filmes_com_links_filtrados.columns]
    ]

    print("\nDataset final criado!")

    return {
        'avaliacoes': avaliacoes_dataset_final,
        'filmes': filmes_info_compacta
    }


def obter_catalogo_filmes_bem_avaliados(dados_processados=None, caminho="./archive"):
    """
    Retorna um DataFrame com filmes únicos que receberam avaliações maiores que 3.
    O resultado é ordenado por popularidade (número de avaliações boas).
    """
    avaliacoes = dados_processados['avaliacoes']
    filmes = dados_processados['filmes']

    # Contar quantas avaliações boas cada filme recebeu
    contagem_avaliacoes = avaliacoes.groupby('movieId').size().reset_index(name='num_avaliacoes')

    # Juntar as informações dos filmes com a contagem de avaliações
    catalogo = pd.merge(
        filmes,
        contagem_avaliacoes,
        on='movieId',
        how='inner'
    )

    # Ordenar por número de avaliações (popularidade) em ordem decrescente
    catalogo = catalogo.sort_values(by='num_avaliacoes', ascending=False)

    # Remover duplicatas caso existam (mantendo a primeira ocorrência - a mais popular)
    catalogo = catalogo.drop_duplicates(subset='movieId')

    # Formatar a data de lançamento, se disponível
    if 'release_date' in catalogo.columns:
        catalogo['release_date'] = pd.to_datetime(catalogo['release_date'], errors='coerce')
        catalogo['release_year'] = catalogo['release_date'].dt.year

    print(f"Catálogo criado com sucesso! {catalogo.shape[0]} filmes únicos bem avaliados.")

    return catalogo

def obter_info_filme(id_filme, filmes):
    """Obtém título e outras informações do filme a partir do DataFrame de filmes"""
    filme = filmes[filmes['id'] == id_filme]
    if not filme.empty:
        return filme.iloc[0]['title']
    return None


def gerar_regras_apriori(dados_processados, suporte_minimo=0.1, confianca_minima=0.1): # Valores de exemplo
    """
    Gera regras de associação
    entre filmes baseado nas avaliações dos usuários.
    """
    avaliacoes = dados_processados['avaliacoes']
    print(f"DEBUG: Número de avaliações recebidas: {len(avaliacoes)}")

    transacoes = {}
    for _, avaliacao in avaliacoes.iterrows():
        usuario_id = avaliacao['userId']
        filme_id = avaliacao['movieId']

        if usuario_id not in transacoes:
            transacoes[usuario_id] = set()

        transacoes[usuario_id].add(filme_id)

    transacoes_lista = list(transacoes.values())
    num_transacoes = len(transacoes_lista)
    if num_transacoes == 0:
        return []

    contagem_itens = {}
    for transacao in transacoes_lista:
        for item in transacao:
            contagem_itens[item] = contagem_itens.get(item, 0) + 1

    itens_frequentes = {item: contagem for item, contagem in contagem_itens.items()
                        if (contagem / num_transacoes) >= suporte_minimo}

    if not itens_frequentes:
        return []

    # Inicializar L1 (conjunto de itens frequentes de tamanho 1)
    L = [{item} for item in itens_frequentes.keys()]

    # Armazenar todos os conjuntos frequentes com seus suportes
    todos_conjuntos_frequentes = {frozenset([item]): contagem / num_transacoes
                                  for item, contagem in itens_frequentes.items()}

    k = 2

    while L:
        Ck = []
        lenL = len(L)
        for i in range(lenL):
            for j in range(i + 1, lenL):
                l1 = sorted(list(L[i]))
                l2 = sorted(list(L[j]))
                if l1[:k-2] == l2[:k-2]:
                    candidato = L[i].union(L[j])
                    if len(candidato) == k:
                        is_valid = True
                        for subconjunto in candidate_subsets(candidato, k - 1):
                             if frozenset(subconjunto) not in todos_conjuntos_frequentes:
                                is_valid = False
                                break
                        if is_valid:
                            Ck.append(candidato)

        Ck_unique = []
        seen_frozen = set()
        for cand in Ck:
            frozen_cand = frozenset(cand)
            if frozen_cand not in seen_frozen:
                Ck_unique.append(cand)
                seen_frozen.add(frozen_cand)
        Ck = Ck_unique

        if not Ck:
            break

        suporte_candidatos = {}
        for transacao in transacoes_lista:
            for candidato_set in Ck:
                if candidato_set.issubset(transacao):
                    candidato_frozen = frozenset(candidato_set)
                    suporte_candidatos[candidato_frozen] = suporte_candidatos.get(candidato_frozen, 0) + 1

        Lk = []
        for candidato_frozen, contagem in suporte_candidatos.items():
            suporte_atual = contagem / num_transacoes
            if suporte_atual >= suporte_minimo:
                Lk.append(set(candidato_frozen))
                todos_conjuntos_frequentes[candidato_frozen] = suporte_atual

        L = Lk
        k += 1

        if k > 5:
             break

    if not todos_conjuntos_frequentes:
        return []

    regras = []

    conjuntos_para_regras = {fs for fs in todos_conjuntos_frequentes if len(fs) >= 2}

    for itemset_frozen in conjuntos_para_regras:
        suporte_itemset = todos_conjuntos_frequentes[itemset_frozen]
        # Gerar todas as regras possíveis a partir deste itemset
        for i in range(1, len(itemset_frozen)):
             # Gera todas as combinações de antecedentes de tamanho 'i'
             from itertools import combinations
             for antecedente_tuple in combinations(itemset_frozen, i):
                antecedente = frozenset(antecedente_tuple)
                consequente = itemset_frozen - antecedente

                if antecedente in todos_conjuntos_frequentes:
                    suporte_antecedente = todos_conjuntos_frequentes[antecedente]
                    if suporte_antecedente > 0: # Evitar divisão por zero
                        confianca = suporte_itemset / suporte_antecedente

                        # Adicionar regra se atender ao critério de confiança mínima
                        if confianca >= confianca_minima:
                             regras.append((list(antecedente), list(consequente), confianca))

    regras_simples = []
    for ant, cons, conf in regras:
        if len(ant) == 1 and len(cons) == 1:
             regras_simples.append((ant[0], cons[0], conf))

    # Ordenar regras por confiança (decrescente)
    regras_simples.sort(key=lambda x: x[2], reverse=True)

    return regras_simples

def candidate_subsets(itemset, size):
    from itertools import combinations
    return [set(subset) for subset in combinations(itemset, size)]


def recomenda_por_ultimo_filme(ultimo_filme_id, regras, dados_filmes):
    """
    Recomenda filmes com base no último filme assistido pelo usuário.
    """
    if not isinstance(ultimo_filme_id, (int, float)):
         return []

    regras_aplicaveis = [regra for regra in regras if regra[0] == ultimo_filme_id]

    if not regras_aplicaveis:
        return []

    recomendacoes = []
    filmes_recomendados = set()

    for antecedente, consequente, confianca in regras_aplicaveis:
        if consequente in filmes_recomendados or consequente == ultimo_filme_id:
            continue

        if 'movieId' not in dados_filmes.columns:
            return []

        filme_info = dados_filmes[dados_filmes['movieId'] == consequente]

        if not filme_info.empty:
            if 'title' in filme_info.columns:
                titulo = filme_info.iloc[0]['title']
            elif 'original_title' in filme_info.columns:
                 titulo = filme_info.iloc[0]['original_title']
            else:
                titulo = f"Título Indisponível (ID: {consequente})"

            recomendacoes.append((consequente, titulo, confianca))
            filmes_recomendados.add(consequente)

    recomendacoes.sort(key=lambda x: x[2], reverse=True)

    return recomendacoes

def interface_usuario():
    """Interface para permitir que o usuário selecione filmes e receba recomendações"""
    # Carrega e pré-processa os dados
    dados_processados = pre_processar_dados()
    regras = gerar_regras_apriori(dados_processados)

    # Lista para armazenar os filmes que o usuário gostou
    filmes_usuario = []
    ultimo_filme = None

    while True:
        print("\n===== SISTEMA DE RECOMENDAÇÃO DE FILMES =====")
        print("1. Ver catálogo de filmes")
        print("2. Buscar filmes")
        print("3. Adicionar filme que gostei")
        print("4. Definir último filme assistido")
        print("5. Ver minha lista de filmes")
        print("6. Obter recomendações baseadas no meu histórico")
        print("7. Obter recomendações baseadas no último filme")
        print("0. Sair")

        escolha = input("\nEscolha uma opção: ")

        if escolha == "1":
            catalogo = obter_catalogo_filmes_bem_avaliados(dados_processados)
            print(f"\nExibindo os 10 filmes mais populares do catálogo:")
            for i, (_, filme) in enumerate(catalogo.head(10).iterrows(), 1):
                titulo = filme['title'] if 'title' in filme else filme['original_title']
                ano = filme['release_year'] if 'release_year' in filme else 'N/A'
                avs = filme['num_avaliacoes']
                id_filme = filme['id']
                print(f"{i}. {titulo} ({ano}) - {avs} avaliações positivas - {id_filme} identificador do filme(id)")

        elif escolha == "4":
            try:
                id_filme = int(input("\nDigite o ID do último filme que você assistiu e gostou: "))
                info_filme = obter_info_filme(id_filme, dados_processados['filmes'])

                if info_filme:
                    ultimo_filme = id_filme
                    if id_filme not in filmes_usuario:
                        filmes_usuario.append(id_filme)
                    print(f"'{info_filme}' definido como seu último filme!")
                else:
                    print("Filme não encontrado. Verifique o ID.")
            except ValueError:
                print("Por favor, digite um número válido.")

        elif escolha == "0":
            print("\nObrigado por usar nosso sistema de recomendação de filmes!")
            break

        elif escolha == "7":
            if not ultimo_filme:
                print("\nVocê precisa definir um último filme assistido primeiro.")
                continue

            recomendacoes = recomenda_por_ultimo_filme(ultimo_filme, regras, dados_processados['filmes'])

            if not recomendacoes:
                print("\nNão foi possível encontrar recomendações com base no seu último filme.")
            else:
                ultimo_info = obter_info_filme(ultimo_filme, dados_processados['filmes'])
                print(f"\n----- RECOMENDAÇÕES COM BASE EM '{ultimo_info}' -----")
                for i, (id_filme, titulo, confianca) in enumerate(recomendacoes[:10], 1):
                    print(f"{i}. {titulo} (ID: {id_filme}, Confiança: {confianca:.2f})")

        else:
            print("\nOpção inválida. Por favor, tente novamente.")


# 4. Função Principal
def principal():
    print("Inicializando sistema de recomendação de filmes...")
    interface_usuario()


if __name__ == "__main__":
    principal()