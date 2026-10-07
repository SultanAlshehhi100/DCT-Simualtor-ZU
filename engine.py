"""The simulator's calculation chain, kept separate from the screen code so it is easy to read and test.

    seats  ->  passengers  ->  point-to-point visitors  ->  hotel check-ins  ->  guest nights
          x load factor    x (1 - transfer - transit)    x check-ins per P2P   x avg length of stay
"""
from dataclasses import dataclass


@dataclass
class Chain:
    seats: float
    pax: float
    p2p_share: float
    p2p: float
    checkins: float
    nights: float


def run_chain(seats, load_factor, transfer_share, transit_share, checkin_yield, avg_stay) -> Chain:
    seats = max(seats, 0.0)
    pax = seats * max(load_factor, 0.0)
    p2p_share = max(0.0, 1.0 - max(transfer_share, 0.0) - max(transit_share, 0.0))
    p2p = pax * p2p_share
    checkins = p2p * max(checkin_yield, 0.0)
    nights = checkins * max(avg_stay, 0.0)
    return Chain(seats, pax, p2p_share, p2p, checkins, nights)
