"""Genera i dati: posizioni di gioco + valori del "maestro".

Il maestro e' una ricerca alfa-beta molto piu' profonda (TEACHER_DEPTH semimosse),
con valutazione a materiale e senza limite di budget. Per ogni posizione s
salviamo il valore che il maestro da' a OGNI mossa legale. Questi valori servono:
  - in predictor.py come target P*(s') del predittore (s' = posizione dopo la mossa);
  - in experiment.ipynb come decisione "corretta" con cui giudicare A, B e C.
Il costo del maestro e' offline: non entra nel budget online del confronto.
"""
import json
import os
import random

from dama import Dama
from search import minimaxA

TEACHER_DEPTH = 7
N_TRAIN, N_TEST = 1500, 500


def sample_position(rng):
    """Qualche mossa di self-play di un giocatore debole e rumoroso
    (ricerca a 2 semimosse, 30% di mosse a caso): posizioni varie ma sensate."""
    board = Dama()
    for _ in range(rng.randint(4, 50)):
        moves = board.legal_moves
        if not moves:
            break
        if rng.random() < 0.3:
            move = rng.choice(moves)
        else:
            values = minimaxA(board, 2)[1]   # valore di ogni mossa a 2 semimosse
            order = list(range(len(moves)))
            rng.shuffle(order)   # i pareggi si risolvono a caso
            move = moves[max(order, key=values.__getitem__)]
        board.push(move)
    return board


def teacher(board):
    """P*: valore di ogni mossa legale secondo la ricerca profonda."""
    values = minimaxA(board, TEACHER_DEPTH)[1]
    return {str(m): v for m, v in zip(board.legal_moves, values)}


def generate(n, seed, seen, need_decision):
    rng = random.Random(seed)
    data = []
    while len(data) < n:
        board = sample_position(rng)
        if board.fen in seen or len(board.legal_moves) < 2:
            continue
        seen.add(board.fen)
        values = teacher(board)
        # nel test teniamo solo posizioni in cui la scelta conta davvero
        if need_decision and max(values.values()) == min(values.values()):
            continue
        data.append({"fen": board.fen, "values": values})
        if len(data) % 50 == 0:
            print(f"  {len(data)}/{n}")
    return data


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    seen = set()   # condiviso: nessuna posizione di test compare nel training
    for name, n, seed, need_decision in [("train", N_TRAIN, 1, False), ("test", N_TEST, 2, True)]:
        print(f"genero {name}...")
        with open(f"data/{name}.json", "w") as f:
            json.dump(generate(n, seed, seen, need_decision), f, indent=1)
