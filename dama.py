"""Il gioco: dama 8x8 della libreria py-draughts, usata cosi' com'e'.

E' la variante "American" (scacchiera 8x8, 12 pezzi per lato, dame che muovono
di una casella). Come nella libreria, la presa e' facoltativa.

Convenzione usata in tutto il progetto: un valore e' sempre dal punto di vista
di CHI DEVE MUOVERE ed e' misurato in "pedine" (+1 = una pedina in piu').
"""
from draughts import AmericanBoard as Dama, Color

KING_VALUE = 2.0        # una dama vale due pedine
WIN = 12 * KING_VALUE   # vittoria = 12 dame: piu' di qualunque materiale a partita non finita

PIECES = ["mie pedine", "mie dame", "sue pedine", "sue dame"]   # i 4 gruppi di feature


def sides(board):
    """Bitboard (mie pedine, mie dame, sue pedine, sue dame) per chi muove."""
    if board.turn == Color.WHITE:
        return board.white_men, board.white_kings, board.black_men, board.black_kings
    return board.black_men, board.black_kings, board.white_men, board.white_kings


def material(board, moves):
    """Valutazione statica del sistema A: pezzi miei - pezzi suoi (dama = 2)."""
    my_men, my_kings, op_men, op_kings = sides(board)
    return (my_men.bit_count() - op_men.bit_count() + KING_VALUE * (my_kings.bit_count() - op_kings.bit_count()))


def board_to_vector(board):
    """phi(s): lo stato cosi' com'e', 4 gruppi x 32 caselle = 128 bit (1 = c'e' il pezzo).

    Gruppi: mie pedine, mie dame, sue pedine, sue dame, dal punto di vista di chi
    muove. Quando muove il nero la scacchiera e' girata (casella sq -> 31 - sq),
    cosi' "avanti" e' sempre la stessa direzione per entrambi i colori.
    """
    white = board.turn == Color.WHITE
    return [(bb >> (sq if white else 31 - sq)) & 1 for bb in sides(board) for sq in range(32)]
