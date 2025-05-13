import pandas as pd
import numpy as np


def carregar_dados():
    """Carrega os dados dos arquivos CSV."""
    avaliacoes = pd.read_csv('../../OneDrive/Documentos/GitHub/manga-player-code-crashers/archive/ratings_small.csv')
    metadados = pd.read_csv('../../OneDrive/Documentos/GitHub/manga-player-code-crashers/archive/movies_metadata.csv', low_memory=False)
    return avaliacoes, metadados


def preprocessar_dados(avaliacoes, metadados):
    """Preprocessa os dados eliminando colunas desnecessárias e corrigindo tipos."""
    avaliacoes = avaliacoes[['userId', 'movieId', 'rating']]
    metadados = metadados[['id', 'title', 'genres']]

    metadados.loc[:, 'id'] = pd.to_numeric(metadados['id'], errors='coerce')
    metadados = metadados.dropna(subset=['id'])
    metadados['id'] = metadados['id'].astype(int)
    metadados.rename(columns={'id': 'movieId'}, inplace=True)

    return avaliacoes, metadados


def unificar_dados(avaliacoes, metadados, avaliacao_minima=3.0):
    """Unifica os dataframes e filtra por avaliação mínima."""
    df = pd.merge(avaliacoes, metadados, on='movieId')
    return df[df['rating'] >= avaliacao_minima]


def criar_dataset(df):
    """Cria mapeamento de usuários para filmes e lista de transações."""
    mapa_usuario_filmes = df.groupby('userId')['title'].apply(list).to_dict()
    return mapa_usuario_filmes, list(mapa_usuario_filmes.values())


def calcular_suporte(conjunto_itens, dataset):
    """Calcula o suporte para um conjunto de itens."""
    itens = set(conjunto_itens)
    total = len(dataset)
    if total == 0:
        return 0.0
    ocorrencias = sum(1 for transacao in dataset
                      if itens.issubset(transacao))
    return ocorrencias / total


def calcular_confianca(X, Y, dataset):
    """Calcula a confiança para uma regra X -> Y."""
    X_set = set(X)
    Y_set = set(Y)

    total_X = sum(1 for t in dataset if X_set.issubset(t))
    if total_X == 0:
        return 0.0

    total_XY = sum(1 for t in dataset
                   if X_set.issubset(t) and Y_set.issubset(t))

    return total_XY / total_X


def calcular_lift(X, Y, dataset):
    """Calcula o lift para uma regra X -> Y."""
    suporte_Y = calcular_suporte(Y, dataset)
    if suporte_Y == 0.0:
        return 0.0

    confianca = calcular_confianca(X, Y, dataset)
    return confianca / suporte_Y


def obter_conjuntos_frequentes_por_nivel(dataset, suporte_minimo, nivel_maximo):
    """Obtém conjuntos de itens frequentes por nível."""
    if not dataset:
        return {}

    total_transacoes = len(dataset)

    transacoes = [set(t) for t in dataset]

    contagem_1 = {}
    for t in transacoes:
        for item in t:
            contagem_1[item] = contagem_1.get(item, 0) + 1

    niveis = {}
    L1 = []
    for item, cnt in contagem_1.items():
        sup = cnt / total_transacoes
        L1.append(((item,), sup))

    itens_freq_1 = [item for item, cnt in contagem_1.items()
                    if (cnt / total_transacoes) >= suporte_minimo]

    niveis[1] = L1

    Lk_itemsets = [tuple([item]) for item in sorted(itens_freq_1)]
    for k in range(2, nivel_maximo + 1):
        Ck = set()
        n = len(Lk_itemsets)
        for i in range(n):
            for j in range(i + 1, n):
                p, q = Lk_itemsets[i], Lk_itemsets[j]
                if p[:-1] == q[:-1]:
                    cand = tuple(sorted(set(p) | set(q)))
                    if len(cand) == k:
                        Ck.add(cand)

        if not Ck:
            break

        Lk = []
        for cand in Ck:
            contador = 0
            for t in transacoes:
                if set(cand).issubset(t):
                    contador += 1
            sup = contador / total_transacoes
            if sup >= suporte_minimo:
                Lk.append((cand, sup))

        if not Lk:
            break

        niveis[k] = Lk

        Lk_itemsets = [itemset for itemset, _ in Lk]

    return niveis


def gerar_permutacoes(conjunto, tamanho):
    """Gera todas as permutações de tamanho específico de um conjunto."""
    if tamanho == 0:
        return [()]

    if tamanho > len(conjunto):
        return []

    resultado = []
    for i, elem in enumerate(conjunto):
        resto = conjunto[:i] + conjunto[i + 1:]
        for perm in gerar_permutacoes(resto, tamanho - 1):
            resultado.append((elem,) + perm)

    return resultado


def gerar_regras_associacao(conjuntos_frequentes, dataset, confianca_minima):
    """Gera regras de associação a partir dos conjuntos frequentes."""
    regras = []
    mapa_regras = {}

    for conjunto, suporte in conjuntos_frequentes:
        if len(conjunto) < 2:
            continue

        itens = list(conjunto)
        n = len(itens)
        full_mask = (1 << n) - 1

        for mask in range(1, full_mask):
            antecedente = [itens[j] for j in range(n) if (mask >> j) & 1]
            consequente = [itens[j] for j in range(n) if not ((mask >> j) & 1)]

            if not consequente:
                continue

            conf = calcular_confianca(antecedente, consequente, dataset)
            if conf < confianca_minima:
                continue

            lft = calcular_lift(antecedente, consequente, dataset)

            ant_tup = tuple(antecedente)
            cons_tup = tuple(consequente)
            regra = (ant_tup, cons_tup, conf, lft)

            regras.append(regra)
            if ant_tup in mapa_regras:
                mapa_regras[ant_tup].append((cons_tup, conf, lft))
            else:
                mapa_regras[ant_tup] = [(cons_tup, conf, lft)]

    return regras, mapa_regras


def recomendar_por_historico(filmes_usuario, mapa_regras, dataset):
    """Recomenda filmes com base no histórico completo do usuário."""
    recomendacoes = {}
    usuario_set = set(filmes_usuario)

    for antecedente, consequentes in mapa_regras.items():
        if set(antecedente) <= usuario_set:
            for consequente, conf, _ in consequentes:
                for f in consequente:
                    if f not in usuario_set and f not in recomendacoes:
                        suporte = calcular_suporte([f], dataset)
                        recomendacoes[f] = (conf, suporte)

    ordenadas = sorted(
        recomendacoes.items(),
        key=lambda item: (item[1][0], item[1][1]),
        reverse=True
    )
    return [filme for filme, _ in ordenadas[:5]]


def recomendar_por_ultimo_filme(filmes_usuario, regras, dataset,
                                limite_confianca_min=0.01, max_tentativas=5):
    """Recomenda filmes com base no último filme assistido pelo usuário."""
    recomendacoes = {}
    usuario_set = set(filmes_usuario)
    tentativas = 0

    for filme in reversed(filmes_usuario):
        tentativas += 1

        for antecedente, consequentes in regras.items():
            for consequente, conf, _ in consequentes:
                if conf < limite_confianca_min:
                    continue
                if filme in antecedente:
                    alvos = consequente
                elif filme in consequente:
                    alvos = antecedente
                else:
                    continue

                for f in alvos:
                    if f not in usuario_set and f not in recomendacoes:
                        suporte = calcular_suporte([f], dataset)
                        recomendacoes[f] = (conf, suporte)

        if recomendacoes or tentativas >= max_tentativas:
            break

    ordenadas = sorted(
        recomendacoes.items(),
        key=lambda item: (item[1][0], item[1][1]),
        reverse=True
    )
    return [filme for filme, _ in ordenadas[:5]]

def recomendar_filmes_populares(filmes_usuario, niveis_frequentes, quantidade=5):
    """Recomenda filmes populares para usuários sem recomendações específicas."""
    nivel1 = niveis_frequentes.get(1, [])
    nivel1.sort(key=lambda x: x[1], reverse=True)

    recomendacoes = []
    for filme_info, _ in nivel1:
        filme = filme_info[0]
        if filme not in filmes_usuario:
            recomendacoes.append(filme)
            if len(recomendacoes) >= quantidade:
                break

    return recomendacoes


def exibir_conjuntos_itens(niveis_frequentes):
    """Exibe os conjuntos de itens frequentes de forma estilizada."""
    df_conjuntos = pd.DataFrame([
        {"Nível": k, "Conjunto de Itens": itemset, "Suporte": round(support, 2)}
        for k, itemsets in niveis_frequentes.items()
        for itemset, support in itemsets
    ])

    print("\n" + "=" * 80)
    print("📊 CONJUNTOS DE ITENS FREQUENTES".center(80))
    print("=" * 80)

    # Configurar opções de exibição para pandas
    with pd.option_context('display.max_rows', 20, 'display.max_columns', None, 'display.width', 1000):
        print(df_conjuntos)

    print("=" * 80)


def exibir_regras(regras, max_exibir=50):
    """
    Exibe as regras de associação de forma estilizada,
    mostrando no máximo as `max_exibir` primeiras regras.
    """
    cabecalhos = ["Antecedente", "Consequente", "Confiança", "Lift"]

    linhas = []
    for antecedente, consequente, confianca, lift in regras:
        linhas.append([
            str(antecedente),
            str(consequente),
            f"{confianca:.2f}",
            f"{lift:.2f}"
        ])

    total_exibir = min(len(linhas), max_exibir)
    linhas = linhas[:total_exibir]

    larguras = []
    for i, cab in enumerate(cabecalhos):
        maior = len(cab)
        for linha in linhas:
            if len(linha[i]) > maior:
                maior = len(linha[i])
        larguras.append(maior)

    total = sum(larguras) + 3 * (len(larguras) - 1)

    separador = "=" * total
    separador_meio = "-" * total

    print("\n" + separador)
    titulo = f"🔗 REGRAS DE ASSOCIAÇÃO (mostrando {total_exibir} de {len(regras)})"
    print(titulo.center(total))
    print(separador)

    formato = " | ".join(f"{{:{w}}}" for w in larguras)

    print(formato.format(*cabecalhos))
    print(separador_meio)
    for linha in linhas:
        print(formato.format(*linha))
    print(separador)

def exibir_recomendacoes(id_usuario, filmes_usuario, recomendacoes_hist, recomendacoes_ult, niveis_frequentes):
    """Exibe as recomendações de filmes para o usuário de forma estilizada."""
    print("\n" + "=" * 80)
    print(f"🎬 FILMES CURTIDOS PELO USUÁRIO {id_usuario}".center(80))
    print("=" * 80)

    for i, filme in enumerate(filmes_usuario, 1):
        print(f"{i}. {filme}")

    print(f"\nTotal de filmes curtidos: {len(filmes_usuario)}")
    print("=" * 80)

    # Recomendações baseadas no histórico
    print("\n" + "=" * 80)
    print("🔍 RECOMENDAÇÕES BASEADAS NO HISTÓRICO COMPLETO".center(80))
    print("=" * 80)

    if recomendacoes_hist:
        for i, filme in enumerate(recomendacoes_hist, 1):
            print(f"{i}. {filme}")
    else:
        recomendacoes_populares = recomendar_filmes_populares(filmes_usuario, niveis_frequentes)
        print("⚠️ Nenhuma recomendação específica encontrada. Sugestões populares:")
        for i, filme in enumerate(recomendacoes_populares, 1):
            print(f"{i}. {filme}")

    print("=" * 80)

    # Recomendações baseadas no último filme
    print("\n" + "=" * 80)
    print("🎯 RECOMENDAÇÕES BASEADAS NO ÚLTIMO FILME CURTIDO".center(80))
    print("=" * 80)

    if recomendacoes_ult:
        for i, filme in enumerate(recomendacoes_ult, 1):
            print(f"{i}. {filme}")
    else:
        recomendacoes_populares = recomendar_filmes_populares(filmes_usuario, niveis_frequentes)
        print("⚠️ Nenhuma recomendação específica encontrada. Sugestões populares:")
        for i, filme in enumerate(recomendacoes_populares, 1):
            print(f"{i}. {filme}")

    print("=" * 80)


def main():
    """Função principal do programa."""
    # Parâmetros
    suporte_minimo = 0.1
    confianca_minima = 0.3
    nivel_maximo = 2

    print("\n" + "=" * 80)
    print("🎥 SISTEMA DE RECOMENDAÇÃO DE FILMES".center(80))
    print("=" * 80)
    print("Por favor aguarde...".center(80))

    avaliacoes, metadados = carregar_dados()
    avaliacoes, metadados = preprocessar_dados(avaliacoes, metadados)
    df = unificar_dados(avaliacoes, metadados)
    mapa_usuario_filmes, dataset = criar_dataset(df)

    print("Carregando regras...".center(80))
    print("=" * 80)

    # Mineração de regras de associação
    niveis_frequentes = obter_conjuntos_frequentes_por_nivel(dataset, suporte_minimo, nivel_maximo)
    todos_conjuntos_frequentes = []
    for nivel in niveis_frequentes.values():
        for item in nivel:
            todos_conjuntos_frequentes.append(item)

    # Exibição de resultados intermediários
    exibir_conjuntos_itens(niveis_frequentes)
    regras, mapa_regras = gerar_regras_associacao(todos_conjuntos_frequentes, dataset, confianca_minima)
    exibir_regras(regras)

    # Interface com o usuário
    print("\n" + "=" * 80)
    print("👤 SELEÇÃO DE USUÁRIO".center(80))
    print("=" * 80)

    try:
        print(f"Existem {len(mapa_usuario_filmes)} usuários disponíveis.")
        id_usuario_indice = int(
            input("Digite o índice do usuário desejado (0-{}): ".format(len(mapa_usuario_filmes) - 1)))
        exemplo_id_usuario = list(mapa_usuario_filmes.keys())[id_usuario_indice]
    except (ValueError, IndexError):
        print("⚠️ ID de usuário inválido. Usando o primeiro usuário como exemplo.")
        exemplo_id_usuario = list(mapa_usuario_filmes.keys())[0]

    filmes_usuario = mapa_usuario_filmes[exemplo_id_usuario]

    # Geração de recomendações
    recomendacoes_hist = recomendar_por_historico(filmes_usuario, mapa_regras, dataset)
    recomendacoes_ult = recomendar_por_ultimo_filme(filmes_usuario, mapa_regras, dataset)

    # Exibição das recomendações
    exibir_recomendacoes(exemplo_id_usuario, filmes_usuario, recomendacoes_hist, recomendacoes_ult, niveis_frequentes)


if __name__ == "__main__":
    main()