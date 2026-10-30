"""Il predittore appreso V_theta (python predictor.py per allenarlo e salvarlo).

Modello: rete neurale di scikit-learn (MLPRegressor) con uno strato nascosto di
64 neuroni; in ingresso phi(s), lo stato (dama.board_to_vector). Impara il valore
che il maestro da' alle posizioni.
"""
import json
import math

import joblib
import numpy as np
from scipy.special import softmax
from sklearn.neural_network import MLPRegressor

from dama import WIN, Dama, board_to_vector
from search import minimaxB

TEMPERATURES = [0.1 * k for k in range(1, 31)]


def from_json(path):
    """Legge i dati del maestro e li trasforma negli esempi (X, y) per scikit-learn.

    Per ogni mossa: X = stato dopo la mossa, y = valore del maestro cambiato di
    segno (dopo la mossa muove l'altro). Ritorna anche le posizioni lette.
    """
    with open(path) as f:
        data = json.load(f)
    X, y = [], []
    for item in data:
        board = Dama.from_fen(item["fen"])
        for m in board.legal_moves:
            child = board.copy()
            child.push(m)
            target = -item["values"][str(m)]
            # posizioni finali e vittorie forzate le riconosce la ricerca, non il predittore
            if child.legal_moves and abs(target) < WIN:
                X.append(board_to_vector(child))
                y.append(target)
    return data, X, y


def calibrate_temperature(model, data):
    """Sceglie la temperatura T di pi = softmax(V/T).

    Tra le T candidate tiene quella che da' piu' probabilita' alle mosse che il
    maestro considera migliori. Cosi' pi(a|s) significa "probabilita' che a sia
    una mossa migliore per il maestro".
    """
    predictions = []   # per ogni posizione: (valori predetti delle mosse, quali sono le migliori)
    for item in data:
        board = Dama.from_fen(item["fen"])
        values = minimaxB(board, 1, model)[1]   # la predizione pura
        best = [item["values"][str(m)] == max(item["values"].values()) for m in board.legal_moves]
        predictions.append((np.array(values), best))
    return max(TEMPERATURES, key=lambda T: sum(math.log(sum(p for p, b in zip(softmax(values / T), best) if b)) for values, best in predictions))


def train(path="data/train.json", out="data/model.joblib"):
    """Carica i dati, allena il modello, calibra la temperatura T di pi e lo salva."""
    data, X, y = from_json(path)
    # early_stopping: tiene da parte il 10% degli esempi e si ferma quando su quelli non migliora
    model = MLPRegressor(hidden_layer_sizes=(64,), early_stopping=True, max_iter=1000, random_state=0).fit(X, y)
    model.T = calibrate_temperature(model, data)
    joblib.dump(model, out)
    return model


if __name__ == "__main__":
    model = train()
    print(f"Rete allenata in {model.n_iter_} epoche, temperatura calibrata T = {model.T:.1f}")
