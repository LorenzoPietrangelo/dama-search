"""La ricerca dei tre sistemi (minimaxA, minimaxB, minimaxC) e il regolatore.

Tutti e tre usano alphabeta, il minimax con potatura alfa-beta. Alla radice
calcolano il valore di ogni mossa (finestra piena), cosi' i valori sono esatti
e si possono confrontare. Ritornano (indice della mossa scelta, valori delle
mosse, nodi esplorati): i nodi di A o di B fanno da limite per C.
"""
import math

import numpy as np
from scipy.special import softmax

from dama import WIN, board_to_vector, material


def alphabeta(board, depth, alpha, beta, evaluate, budget):
    """Minimax con potatura alfa-beta (forma negamax): valore per chi muove in `board`.

    budget = {"used": nodi esplorati, "limit": massimo, "stopped": limite raggiunto?}.
    Ogni mossa eseguita conta 1 nodo; al limite la ricerca si interrompe.
    """
    moves = board.legal_moves
    if not moves:
        return -WIN                     # chi non puo' muovere ha perso
    if depth == 0:
        return evaluate(board, moves)   # orizzonte: qui la ricerca esplicita si ferma
    best = -math.inf
    for move in moves:
        if budget["used"] >= budget["limit"]:
            budget["stopped"] = True
            return best                 # limite raggiunto: il valore non conta piu'
        budget["used"] += 1
        board.push(move)
        value = -alphabeta(board, depth - 1, -beta, -alpha, evaluate, budget)
        board.pop()
        best = max(best, value)
        alpha = max(alpha, value)
        if alpha >= beta:
            break
    return best


def minimaxA(board, depth):
    """Sistema A: ogni mossa cercata a `depth` semimosse, foglie valutate col materiale."""
    budget = {"used": 0, "limit": math.inf, "stopped": False}
    values = []
    for move in board.legal_moves:
        budget["used"] += 1
        board.push(move)
        values.append(-alphabeta(board, depth - 1, -math.inf, math.inf, material, budget))
        board.pop()
    return values.index(max(values)), values, budget["used"]


def minimaxB(board, depth, model):
    """Sistema B: come A, ma alle foglie c'e' il predittore, che stima cosa vedrebbe
    una ricerca piu' profonda: e' come estendere virtualmente la profondita'."""
    evaluate = lambda b, moves: model.predict([board_to_vector(b)])[0]
    budget = {"used": 0, "limit": math.inf, "stopped": False}
    values = []
    for move in board.legal_moves:
        budget["used"] += 1
        board.push(move)
        values.append(-alphabeta(board, depth - 1, -math.inf, math.inf, evaluate, budget))
        board.pop()
    return values.index(max(values)), values, budget["used"]


def regulate(values, T, rho_max=0.05):
    """Il regolatore: valori attuali delle mosse -> frontiera attiva. Ritorna (mosse attive, rho).

    Tiene le mosse piu' probabili secondo pi = softmax(V/T), dalla piu' probabile,
    finche' insieme coprono 1 - rho_max della probabilita'. rho e' la probabilita'
    rimasta sulle mosse escluse.
    """
    pi = softmax(np.array(values) / T)
    active, covered = [], 0.0
    for i in sorted(range(len(values)), key=lambda i: -pi[i]):
        if covered >= 1 - rho_max:
            break
        active.append(i)
        covered += pi[i]
    return active, 1 - covered


def minimaxC(board, depth, model, max_nodes, T=None, rho_max=0.05):
    """Sistema C: come B, piu' il regolatore e un limite di `max_nodes` nodi.

    Prima ogni mossa riceve la predizione (profondita' 1). Poi si cercano a
    `depth` solo le mosse della frontiera attiva, dalla piu' probabile, una alla
    volta; dopo ogni mossa il regolatore ricalcola la frontiera con il valore
    appena trovato. Le mosse non cercate tengono la predizione.
    Ritorna anche rho, la probabilita' lasciata sulle mosse escluse.
    """
    evaluate = lambda b, moves: model.predict([board_to_vector(b)])[0]
    T = T or model.T
    budget = {"used": 0, "limit": max_nodes, "stopped": False}
    moves = board.legal_moves
    values = []
    for move in moves:                  # la predizione di ogni mossa
        budget["used"] += 1
        board.push(move)
        values.append(-alphabeta(board, 0, -math.inf, math.inf, evaluate, budget))
        board.pop()
    searched = [False] * len(moves)
    while True:
        active, rho = regulate(values, T, rho_max)   # frontiera, con i valori trovati finora
        todo = [i for i in active if not searched[i]]
        if not todo or budget["used"] >= budget["limit"]:
            break
        i = todo[0]                     # la piu' probabile tra quelle non ancora cercate
        board.push(moves[i])            # gia' contata nella predizione: non si paga due volte
        value = -alphabeta(board, depth - 1, -math.inf, math.inf, evaluate, budget)
        board.pop()
        if budget["stopped"]:
            break                       # limite raggiunto durante questa mossa: valore non valido
        values[i], searched[i] = value, True
    return values.index(max(values)), values, budget["used"], rho
