#!/usr/bin/env python3
"""Metodologie computațională — distanțe fonetice / sintactice (extras din document)."""
import numpy as np
from scipy.spatial.distance import cityblock

# 1. MATRICEA DE CLASE FONETICE ȘI COSTURI DE SUBSTITUȚIE (Stil ASJP)
SOUND_CLASSES = {
    # Labiale / Labiodentale
    "p": "p", "b": "p", "f": "p", "v": "p",
    # Dentale / Alveolare
    "t": "t", "d": "t", "s": "t", "z": "t",
    # Velare / Labiovelare
    "k": "k", "g": "k", "q": "k", "x": "k",
    # Nazale
    "m": "n", "n": "n",
    # Lichide
    "r": "l", "l": "l",
    # Vocale
    "a": "v", "e": "v", "i": "v", "o": "v", "u": "v", "ă": "v", "î": "v",
}


def get_phonetic_cost(char1, char2):
    """Calculează costul de substituție între două caractere."""
    if char1 == char2:
        return 0.0
    class1 = SOUND_CLASSES.get(char1, char1)
    class2 = SOUND_CLASSES.get(char2, char2)
    if class1 == class2:
        return 0.4  # aceeași clasă
    # Cazul special: velară/labiovelară (k/q) → labială (p)
    if (class1 == "k" and class2 == "p") or (class1 == "p" and class2 == "k"):
        return 0.5
    return 1.0


def weighted_levenshtein(s1, s2):
    """Distanța de editare pe baza costurilor fonetice."""
    m, n = len(s1), len(s2)
    dp = np.zeros((m + 1, n + 1))
    for i in range(m + 1):
        dp[i, 0] = i * 0.8
    for j in range(n + 1):
        dp[0, j] = j * 0.8
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = get_phonetic_cost(s1[i - 1], s2[j - 1])
            dp[i, j] = min(
                dp[i - 1, j] + 0.8,      # ștergere
                dp[i, j - 1] + 0.8,      # inserție
                dp[i - 1, j - 1] + cost,  # substituție
            )
    return dp[m, n]


def normalized_proximity(s1, s2):
    """Transformă distanța în proximitate (0 = diferit, 1 = identic)."""
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    dist = weighted_levenshtein(s1, s2)
    return max(0.0, 1.0 - (dist / (max_len * 0.8)))


def permutation_test(romanian_words, source_words, num_permutations=1000):
    """Verifică dacă alinierea e reală sau ambiguă (detecție p-hacking)."""
    observed_proxed = [
        normalized_proximity(ro, src) for ro, src in zip(romanian_words, source_words)
    ]
    observed_mean = float(np.mean(observed_proxed))

    null_distributions = []
    words_copy = list(source_words)
    np.random.seed(42)
    for _ in range(num_permutations):
        np.random.shuffle(words_copy)
        perm_proxed = [
            normalized_proximity(ro, src)
            for ro, src in zip(romanian_words, words_copy)
        ]
        null_distributions.append(float(np.mean(perm_proxed)))

    p_value = float(np.sum(np.array(null_distributions) >= observed_mean) / num_permutations)
    return observed_mean, p_value


FEATURES = [
    "Articol enclitic",
    "Înlocuire infinitiv",
    "Viitor cu vrea",
    "Genitiv-Dativ identic",
    "Vocativ în -o",
    "Pronume posesiv antepus",
    "Reduplicare clitică",
    "Numerale 11-19 ca 'peste zece'",
    "Prepoziție 'la' pt acuzativ",
    "Sufix -mânt",
    "Absența neutrului morfologic pur",
    "Sincretism Locativ-Ablativ",
]

languages_syntax = {
    "Română": np.array([1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1]),
    "Albaneză": np.array([1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1]),
    "Bulgară": np.array([1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1]),
    "Latină": np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0]),
    "Franceză": np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
}


def calculate_manhattan_distances():
    print("\n=== DISTANȚA MANHATTAN NORMALIZATĂ (Divergență Structurală) ===")
    base_lang = "Română"
    for lang, vec in languages_syntax.items():
        if lang == base_lang:
            continue
        dist = cityblock(languages_syntax[base_lang], vec) / len(FEATURES)
        status = "SIMILARĂ" if dist < 0.3 else "DIVERGENTĂ"
        print(f"D_typ({base_lang}, {lang}) = {dist:.2f} (Structură {status})")


if __name__ == "__main__":
    ro_lexicon = ["apă", "noapte", "patru", "iepure"]
    lat_lexicon = ["aqua", "noctem", "quattuor", "leprem"]

    print("=== TEST 1: ALINIERE FONETICĂ PONDERATĂ ===")
    for ro, lat in zip(ro_lexicon, lat_lexicon):
        prox = normalized_proximity(ro, lat)
        print(f"Potrivire fonetică Română '{ro}' <-> Latină '{lat}': {prox:.2f}")

    print("\n=== TEST 2: TEST DE PERMUTARE (DETECȚIE P-HACKING) ===")
    obs_mean, p_val = permutation_test(ro_lexicon, lat_lexicon)
    print(f"Proximitatea medie observată: {obs_mean:.4f}")
    print(f"Valoarea p: {p_val:.4f}")
    if p_val > 0.05:
        print("-> [CRITICAL] Semnal ambiguu statistic (posibil p-hacking).")
    else:
        print("-> Semnalul lexical rezistă la amestecare aleatorie.")

    calculate_manhattan_distances()
