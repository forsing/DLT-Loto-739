#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
    - pojednostavljena tabularna mreža sa pažnjom;
    - dvosmerni, dvoslojni LSTM sa pažnjom;
    - dvoslojni Transformer sa četiri glave pažnje;
    - neuronski ansambl: 25% / 35% / 40%;
    - više strategija za predlaganje kandidata;
    - model istorijskih obrazaca celih kombinacija;
    - elitizam, ukrštanje i mutacije kandidata;
    - zajedničko rangiranje završnih kombinacija.

    - jedna zona 7/39;
    - tri nezavisna CSV-a;
    - regresija distribucije; 
    - distribucioni opisi umesto rangiranja frekvencija brojeva;
    - determinističke operacije umesto random postupaka;
    - hronološka validacija;
    - bez ručnih pravila za paritet, polovine, razmake i nizove.

CSV:
    Bez zaglavlja.
    Sedam različitih celih brojeva 1..39 u svakom redu.
    Najstariji red prvi, najnoviji poslednji.

Biblioteke:
    numpy
    torch

Pretraga kandidata je deterministička i heuristička:
ne predstavlja iscrpnu pretragu svih kombinacija.
"""


import argparse
import csv
import hashlib
import math
import sys
from pathlib import Path

try:
    import numpy as np
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError as exc:
    raise SystemExit(
        "Potrebne biblioteke: numpy i torch.\n"
        "Instalacija: python3 -m pip install numpy torch\n"
        f"Detalj: {exc}"
    )


DEFAULT_FILES = (
    "/data/loto7_4704_k83.csv",
    "data/loto7_4704_k83_loto_2974.csv",
    "/data/loto7_4704_k83_loto_plus_1730.csv",
)

SEQUENCE = 20
VALIDATION = 48
TEST = 64
MIN_ROWS = 300

MAX_EPOCHS = 30
PATIENCE = 6
BATCH_SIZE = 128

BEAM_WIDTH = 20
POPULATION = 30
GENERATIONS = 6
ELITES = 3

MODEL_WEIGHTS = np.array([0.25, 0.35, 0.40], dtype=float)
MODEL_NAMES = ("Tabular_attention", "BiLSTM", "Transformer")

GRID = np.arange(1, 40)
PAIR_7 = np.triu_indices(7, 1)
PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37}

TINY = np.finfo(np.float64).tiny


def load_csv(filename):
    path = Path(filename).expanduser().resolve(strict=True)
    rows = []

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for line_number, row in enumerate(csv.reader(handle), 1):
            if not row or all(not cell.strip() for cell in row):
                continue

            if len(row) != 7:
                raise ValueError(
                    f"{path.name}, red {line_number}: "
                    f"očekivano 7 kolona; dobijeno {len(row)}."
                )

            try:
                numbers = [int(cell.strip()) for cell in row]
            except ValueError as exc:
                raise ValueError(
                    f"{path.name}, red {line_number}: "
                    "sve vrednosti moraju biti celi brojevi."
                ) from exc

            if len(set(numbers)) != 7:
                raise ValueError(
                    f"{path.name}, red {line_number}: ponovljen broj."
                )

            if min(numbers) < 1 or max(numbers) > 39:
                raise ValueError(
                    f"{path.name}, red {line_number}: "
                    "broj izvan opsega 1..39."
                )

            rows.append(sorted(numbers))

    if len(rows) < MIN_ROWS:
        raise ValueError(
            f"{path.name}: potrebno najmanje {MIN_ROWS} izvlačenja."
        )

    draws = np.asarray(rows, dtype=np.int64)
    canonical = "\n".join(
        ",".join(map(str, row)) for row in draws
    ).encode("ascii")

    return path, draws, hashlib.sha256(canonical).hexdigest()


def draw_profile(draw):
    values = np.asarray(draw, dtype=float)
    distances = np.abs(values[:, None] - values[None, :])[PAIR_7]

    return np.r_[
        values / 39.0,
        [np.mean(distances <= threshold)
         for threshold in (2, 5, 10, 20)],
        (values[-1] - values[0]) / 38.0,
    ]


def pattern_profile(combinations, previous):
    """
    Opisuje cele kombinacije.
    Nijedna osobina nema unapred propisanu 'poželjnu' vrednost.
    """
    combinations = np.asarray(combinations, dtype=np.int64)
    previous = np.asarray(previous, dtype=np.int64)

    if combinations.ndim == 1:
        combinations = combinations[None, :]

    if previous.ndim == 1:
        previous = np.broadcast_to(previous, combinations.shape)

    if previous.shape != combinations.shape:
        raise ValueError("Neusklađeni prethodni redovi za opis obrazaca.")

    gaps = np.diff(combinations, axis=1)
    distances = np.abs(
        combinations[:, :, None] - combinations[:, None, :]
    )[:, PAIR_7[0], PAIR_7[1]]

    overlap = (
        combinations[:, :, None] == previous[:, None, :]
    ).any(axis=2).sum(axis=1)

    unique_tails = np.array([
        len(set((row % 10).tolist())) for row in combinations
    ], dtype=float)

    arithmetic_complexity = np.array([
        (len(np.unique(row)) - 6) / 15.0
        for row in distances
    ])

    prime_count = np.array([
        sum(int(number) in PRIMES for number in row)
        for row in combinations
    ], dtype=float)

    return np.column_stack([
        (combinations[:, -1] - combinations[:, 0]) / 38.0,
        combinations.sum(axis=1) / 273.0,
        (gaps == 1).sum(axis=1) / 6.0,
        (combinations % 2 == 1).sum(axis=1) / 7.0,
        overlap / 7.0,
        unique_tails / 7.0,
        (combinations <= 13).sum(axis=1) / 7.0,
        ((combinations >= 14) & (combinations <= 26)).sum(axis=1) / 7.0,
        (combinations >= 27).sum(axis=1) / 7.0,
        prime_count / 7.0,
        arithmetic_complexity,
    ]).astype(float)


def context_summary(sequences):
    sequences = np.asarray(sequences)
    return np.concatenate([
        sequences[:, -1],
        sequences.mean(axis=1),
        sequences.std(axis=1),
        sequences[:, -1] - sequences[:, 0],
    ], axis=1)


def make_dataset(draws):
    profiles = np.asarray(
        [draw_profile(draw) for draw in draws],
        dtype=np.float32,
    )

    sequences = np.stack([
        profiles[t - SEQUENCE:t]
        for t in range(SEQUENCE, len(draws) + 1)
    ])

    targets = draws[SEQUENCE:]

    observed_cdf = (
        targets[:, :, None] <= GRID[None, None, :]
    ).astype(np.float32)

    previous = draws[SEQUENCE - 1:-1]
    patterns = pattern_profile(targets, previous)

    return sequences, targets, observed_cdf, patterns


def fit_scaler(x):
    mean = x.mean(axis=0, dtype=np.float64)
    scale = x.std(axis=0, dtype=np.float64)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return mean, scale


def analytic_fill(parameter, phase):
    """Analitička, neslučajna inicijalizacija."""
    with torch.no_grad():
        if parameter.ndim < 2:
            parameter.zero_()
            return

        rows = parameter.shape[0]
        columns = parameter.numel() // rows

        r = torch.arange(
            1, rows + 1, dtype=torch.float64
        )[:, None]

        c = torch.arange(
            1, columns + 1, dtype=torch.float64
        )[None, :]

        values = torch.sin(
            r * c * math.sqrt(2.0)
            + 0.113 * r.square()
            + 0.071 * c.square()
            + phase
        )

        values *= math.sqrt(2.0 / columns)

        parameter.copy_(
            values.reshape(parameter.shape).to(parameter.dtype)
        )


class Linear(nn.Module):
    def __init__(self, inputs, outputs, phase):
        super().__init__()

        self.weight = nn.Parameter(torch.empty(outputs, inputs))
        self.bias = nn.Parameter(torch.zeros(outputs))

        analytic_fill(self.weight, phase)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class DeterministicLSTM(nn.LSTM):
    def reset_parameters(self):
        # nn.LSTM poziva ovaj metod tokom svog konstruktora.
        # Standardna slučajna inicijalizacija je time zamenjena.
        for index, (name, parameter) in enumerate(self.named_parameters()):
            analytic_fill(parameter, 0.29 + 0.19 * index)

            if "bias" in name:
                with torch.no_grad():
                    hidden = parameter.numel() // 4
                    parameter[hidden:2 * hidden].fill_(0.5)


class DistributionHeads(nn.Module):
    def __init__(self, inputs, phase):
        super().__init__()

        self.heads = nn.ModuleList([
            Linear(inputs, 39, phase + position * 0.17)
            for position in range(7)
        ])

    def forward(self, x):
        logits = torch.stack(
            [head(x) for head in self.heads], dim=1
        )

        # Pozitivni intenziteti diskretne distribucije.
        # Uči se CDF regresijom; ne koristi se cross-entropy.
        intensity = F.softplus(logits.to(torch.float64))
        intensity = intensity.clamp_min(torch.finfo(torch.float64).tiny)

        return intensity / intensity.sum(dim=-1, keepdim=True)


class TabularAttention(nn.Module):
    def __init__(self, features):
        super().__init__()
        inputs = features * 4

        self.fc1 = Linear(inputs, 64, 0.13)
        self.norm1 = nn.LayerNorm(64)
        self.fc2 = Linear(64, 64, 0.31)
        self.norm2 = nn.LayerNorm(64)

        self.attention = Linear(64, inputs, 0.47)
        self.combine = Linear(64 + inputs, 32, 0.61)
        self.output = DistributionHeads(32, 0.83)

    def forward(self, sequence):
        tabular = torch.cat([
            sequence[:, -1],
            sequence.mean(dim=1),
            sequence.std(dim=1, unbiased=False),
            sequence[:, -1] - sequence[:, 0],
        ], dim=1)

        hidden = F.relu(self.norm1(self.fc1(tabular)))
        hidden = F.relu(self.norm2(self.fc2(hidden)))

        weights = F.softmax(self.attention(hidden), dim=-1)
        combined = torch.cat([hidden, tabular * weights], dim=-1)

        return self.output(F.relu(self.combine(combined)))


class LSTMNetwork(nn.Module):
    def __init__(self, features):
        super().__init__()

        self.project = Linear(features, 32, 0.19)

        self.lstm = DeterministicLSTM(
            input_size=32,
            hidden_size=32,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.0,
        )

        self.attention = Linear(64, 1, 0.43)
        self.combine = Linear(64, 32, 0.67)
        self.output = DistributionHeads(32, 0.97)

    def forward(self, sequence):
        hidden, _ = self.lstm(self.project(sequence))
        attention = F.softmax(self.attention(hidden), dim=1)
        context = (hidden * attention).sum(dim=1)

        return self.output(F.relu(self.combine(context)))


class AttentionBlock(nn.Module):
    def __init__(self, width, heads, phase):
        super().__init__()

        self.width = width
        self.heads = heads
        self.head_width = width // heads

        self.q = Linear(width, width, phase)
        self.k = Linear(width, width, phase + 0.11)
        self.v = Linear(width, width, phase + 0.23)
        self.mix = Linear(width, width, phase + 0.37)

        self.norm1 = nn.LayerNorm(width)
        self.ff1 = Linear(width, width * 4, phase + 0.53)
        self.ff2 = Linear(width * 4, width, phase + 0.71)
        self.norm2 = nn.LayerNorm(width)

    def forward(self, x):
        batch, steps, _ = x.shape

        def split(value):
            return value.reshape(
                batch, steps, self.heads, self.head_width
            ).transpose(1, 2)

        q = split(self.q(x))
        k = split(self.k(x))
        v = split(self.v(x))

        attention = F.softmax(
            (q @ k.transpose(-2, -1)) / math.sqrt(self.head_width),
            dim=-1,
        )

        context = (attention @ v).transpose(1, 2).reshape(
            batch, steps, self.width
        )

        x = self.norm1(x + self.mix(context))
        return self.norm2(x + self.ff2(F.relu(self.ff1(x))))


class TransformerNetwork(nn.Module):
    def __init__(self, features):
        super().__init__()
        width = 32

        self.project = Linear(features, width, 0.17)
        self.blocks = nn.ModuleList([
            AttentionBlock(width, 4, 0.41),
            AttentionBlock(width, 4, 1.13),
        ])
        self.output = DistributionHeads(width, 1.91)

        positions = torch.arange(
            SEQUENCE, dtype=torch.float32
        )[:, None]

        frequencies = torch.exp(
            torch.arange(0, width, 2, dtype=torch.float32)
            * (-math.log(10000.0) / width)
        )

        encoding = torch.zeros(SEQUENCE, width)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        encoding[:, 1::2] = torch.cos(positions * frequencies)

        self.register_buffer("encoding", encoding[None, :, :])

    def forward(self, sequence):
        x = self.project(sequence) * math.sqrt(32.0)
        x = x + self.encoding[:, :sequence.shape[1]]

        for block in self.blocks:
            x = block(x)

        return self.output(x[:, -1])


NETWORKS = {
    "Tabular_attention": TabularAttention,
    "BiLSTM": LSTMNetwork,
    "Transformer": TransformerNetwork,
}


class NeuralBranch:
    def __init__(self, name):
        self.name = name

    def transform(self, x):
        return np.ascontiguousarray(
            (x - self.mean) / self.scale,
            dtype=np.float32,
        )

    def fit(self, x, cdf, epochs, validation=None):
        self.mean, self.scale = fit_scaler(
            x.reshape(-1, x.shape[-1])
        )

        features = torch.from_numpy(self.transform(x))
        target = torch.from_numpy(cdf.astype(np.float64))

        self.model = NETWORKS[self.name](x.shape[-1])

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=0.001,
            weight_decay=1e-5,
        )

        if validation is not None:
            vx, vy = validation
            valid_features = torch.from_numpy(self.transform(vx))
            valid_target = torch.from_numpy(vy.astype(np.float64))

        best_loss = float("inf")
        best_state = None
        best_epoch = epochs
        stale = 0

        for epoch in range(1, epochs + 1):
            self.model.train()

            for start in range(0, len(features), BATCH_SIZE):
                end = min(start + BATCH_SIZE, len(features))

                pmf = self.model(features[start:end])
                predicted_cdf = pmf.cumsum(dim=-1)
                loss = F.mse_loss(predicted_cdf, target[start:end])

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 1.0
                )
                optimizer.step()

            if validation is not None:
                self.model.eval()

                with torch.no_grad():
                    current = float(F.mse_loss(
                        self.model(valid_features).cumsum(dim=-1),
                        valid_target,
                    ).item())

                if not math.isfinite(current):
                    raise FloatingPointError(
                        f"{self.name}: nevažeći validacioni gubitak."
                    )

                if current < best_loss - 1e-8:
                    best_loss = current
                    best_epoch = epoch
                    stale = 0
                    best_state = {
                        key: value.detach().clone()
                        for key, value in self.model.state_dict().items()
                    }
                else:
                    stale += 1

                if stale >= PATIENCE:
                    break

            if epoch % 5 == 0:
                print(
                    f"    {self.name}: epoha {epoch}/{epochs}",
                    flush=True,
                )

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.model.eval()
        self.epochs = best_epoch
        return self

    def predict(self, x):
        features = torch.from_numpy(self.transform(x))
        result = []

        self.model.eval()

        with torch.no_grad():
            for start in range(0, len(features), BATCH_SIZE):
                result.append(
                    self.model(
                        features[start:start + BATCH_SIZE]
                    ).cpu().numpy()
                )

        output = np.concatenate(result, axis=0)

        if not np.all(np.isfinite(output)) or np.any(output <= 0):
            raise FloatingPointError(
                f"{self.name}: nevažeća distribucija."
            )

        return output


class PatternModel:
    """
    Uslovna kernel-distribucija istorijskih obrazaca.

    Koristi sve trening prelaze:
    sličnost konteksta određuje težine celih istorijskih kombinacija.
    Ne koristi unapred zadate ciljeve za sumu, paritet, zone ili nizove.
    """

    def fit(self, sequences, patterns, target_draws):
        context = context_summary(sequences)
        self.context_mean, self.context_scale = fit_scaler(context)

        self.context = (
            context - self.context_mean
        ) / self.context_scale

        self.pattern_mean, self.pattern_scale = fit_scaler(patterns)

        self.patterns = (
            patterns - self.pattern_mean
        ) / self.pattern_scale

        self.pattern_norm = np.sum(self.patterns ** 2, axis=1)
        self.target_draws = target_draws.copy()
        return self

    def condition(self, sequence):
        context = context_summary(sequence[None, :, :])[0]
        query = (
            context - self.context_mean
        ) / self.context_scale

        distance = np.mean(
            (self.context - query) ** 2,
            axis=1,
        )

        positive = distance[distance > 1e-12]
        bandwidth = float(np.median(positive)) if len(positive) else 1.0

        weights = (
            1.0 + distance / max(bandwidth, 1e-12)
        ) ** -2.0

        weights /= weights.sum()

        # Samo predlaganje početnih kandidata koristi najbliže primere.
        # Ocena kandidata i dalje koristi sve trening prelaze.
        nearest = np.lexsort((
            np.arange(len(distance)), distance
        ))[:16]

        seeds = [
            tuple(map(int, row))
            for row in self.target_draws[nearest]
        ]

        return weights, seeds

    def score(self, combinations, previous, weights):
        raw = pattern_profile(combinations, previous)
        z = (raw - self.pattern_mean) / self.pattern_scale

        output = []

        for start in range(0, len(z), 128):
            batch = z[start:start + 128]

            distance = (
                np.sum(batch ** 2, axis=1)[:, None]
                + self.pattern_norm[None, :]
                - 2.0 * batch @ self.patterns.T
            )

            distance = np.maximum(
                distance / self.patterns.shape[1], 0.0
            )

            similarity = (1.0 + distance) ** -2.0
            output.append(similarity @ weights)

        return np.concatenate(output)


def log_mixture(log_scores):
    top = np.max(log_scores, axis=1)
    return top + np.log(
        np.exp(log_scores - top[:, None]) @ MODEL_WEIGHTS
    )


def proposal_beam(log_pmf, strategy):
    """
    Pet bazena:
        0: tabularna mreža
        1: LSTM
        2: Transformer
        3: ponderisani neuronski konsenzus
        4: minimax saglasnost modela

    Šesti bazen nastaje evolutivnim poboljšavanjem.
    Istorijski obrasci daju dodatne početne predloge.
    """
    beam = [((), np.zeros(3))]

    for position in range(7):
        expanded = []

        for prefix, accumulated in beam:
            last = prefix[-1] if prefix else 0

            for number in range(last + 1, 34 + position):
                values = (
                    accumulated + log_pmf[:, position, number - 1]
                )

                if strategy < 3:
                    score = float(values[strategy])
                elif strategy == 3:
                    score = float(values @ MODEL_WEIGHTS)
                else:
                    score = float(values.min())

                expanded.append((
                    score,
                    prefix + (number,),
                    values,
                ))

        expanded.sort(key=lambda item: (-item[0], item[1]))

        beam = [
            (prefix, values)
            for _, prefix, values in expanded[:BEAM_WIDTH]
        ]

    return [prefix for prefix, _ in beam]


class CandidateScorer:
    def __init__(self, pmf, pattern_model, sequence, previous):
        self.log_pmf = np.log(pmf)
        self.pattern_model = pattern_model
        self.previous = previous
        self.weights, self.pattern_seeds = pattern_model.condition(sequence)
        self.cache = {}

    def scores(self, candidates):
        candidates = sorted(set(tuple(item) for item in candidates))
        missing = [item for item in candidates if item not in self.cache]

        if missing:
            values = np.asarray(missing, dtype=np.int64)

            # (kandidati, modeli, sortirane pozicije)
            selected = np.stack([
                self.log_pmf[:, position, values[:, position] - 1].T
                for position in range(7)
            ], axis=-1)

            joint_log = selected.sum(axis=-1)

            voting = joint_log @ MODEL_WEIGHTS
            mixture = log_mixture(joint_log)
            minimax = joint_log.min(axis=1)

            # Tri načina fuzije procenjenih distribucija.
            neural = (
                0.50 * mixture
                + 0.30 * voting
                + 0.20 * minimax
            ) / 7.0

            pattern = self.pattern_model.score(
                values, self.previous, self.weights
            )

            # Zadržana uloga neuronske ocene od 25%;
            # ostalo je naučena uslovna distribucija obrazaca.
            final = (
                0.75 * np.log(np.maximum(pattern, TINY))
                + 0.25 * neural
            )

            if not np.all(np.isfinite(final)):
                raise FloatingPointError("Nevažeća ocena kandidata.")

            for item, score in zip(missing, final):
                self.cache[item] = float(score)

        return np.asarray([self.cache[item] for item in candidates])

    def rank(self, candidates):
        candidates = sorted(set(tuple(item) for item in candidates))
        scores = self.scores(candidates)

        return [
            candidates[index]
            for index in sorted(
                range(len(candidates)),
                key=lambda index: (-scores[index], candidates[index]),
            )
        ]


def crossover_candidates(parents):
    children = set()

    for first_index, first in enumerate(parents):
        for second in parents[first_index + 1:]:
            union = sorted(set(first) | set(second))

            for cut in range(1, 7):
                for left, right in ((first, second), (second, first)):
                    child = list(dict.fromkeys(
                        left[:cut] + right[cut:]
                    ))

                    for number in union:
                        if len(child) >= 7:
                            break
                        if number not in child:
                            child.append(number)

                    if len(child) == 7:
                        children.add(tuple(sorted(child)))

    return children


def mutation_candidates(parents):
    children = set()

    for parent in parents:
        present = set(parent)

        for removed in parent:
            remaining = present - {removed}

            for added in range(1, 40):
                if added in present:
                    continue
                children.add(tuple(sorted(remaining | {added})))

    return children


def select_next(pmf, pattern_model, sequence, previous):
    if pmf.shape != (3, 7, 39):
        raise ValueError("Neispravan oblik neuronskog ansambla.")

    scorer = CandidateScorer(
        pmf, pattern_model, sequence, previous
    )

    candidates = set(scorer.pattern_seeds)

    for strategy in range(5):
        candidates.update(
            proposal_beam(scorer.log_pmf, strategy)
        )

    population = scorer.rank(candidates)[:POPULATION]

    for _ in range(GENERATIONS):
        elite = population[:ELITES]

        children = set(population)
        children.update(crossover_candidates(population[:8]))
        children.update(mutation_candidates(elite))

        updated = scorer.rank(children)[:POPULATION]

        if updated == population:
            break

        population = updated

    winner = population[0]

    if (
        len(winner) != 7
        or len(set(winner)) != 7
        or min(winner) < 1
        or max(winner) > 39
    ):
        raise ValueError("Završna kombinacija nije validna za 7/39.")

    return winner, scorer.cache[winner], len(scorer.cache)


def fit_ensemble(x, cdf, epochs):
    branches = []

    for name in MODEL_NAMES:
        print(f"  Treniranje: {name}", flush=True)

        branches.append(
            NeuralBranch(name).fit(
                x, cdf, epochs=epochs[name]
            )
        )

    return branches


def predict_ensemble(branches, sequences):
    # Izlaz: (broj konteksta, tri modela, sedam pozicija, 39 brojeva).
    return np.stack([
        branch.predict(sequences)
        for branch in branches
    ], axis=1)


def longest_run(combination):
    current = maximum = 1

    for previous, number in zip(combination[:-1], combination[1:]):
        current = current + 1 if number == previous + 1 else 1
        maximum = max(maximum, current)

    return maximum


def process_dataset(path, draws, fingerprint):
    sequences, targets, cdf, patterns = make_dataset(draws)

    count = len(targets)
    test_start = count - TEST
    validation_start = test_start - VALIDATION

    print("\n" + "=" * 76)
    print(path.name)
    print(f"Izvlačenja: {len(draws)}")
    print(f"Prelaza za finalno treniranje: {count}")
    print(f"SHA-256: {fingerprint}")
    print(
        "Validacija — CSV redovi: "
        f"{SEQUENCE + validation_start + 1}.."
        f"{SEQUENCE + test_start}"
    )
    print(
        "Završna provera — CSV redovi: "
        f"{SEQUENCE + test_start + 1}..{len(draws)}"
    )

    print("\n1. Hronološki izbor dužine treniranja", flush=True)
    epochs = {}

    for name in MODEL_NAMES:
        print(f"  Validacija: {name}", flush=True)

        branch = NeuralBranch(name).fit(
            sequences[:validation_start],
            cdf[:validation_start],
            epochs=MAX_EPOCHS,
            validation=(
                sequences[validation_start:test_start],
                cdf[validation_start:test_start],
            ),
        )

        epochs[name] = branch.epochs
        print(f"    Izabrano epoha: {branch.epochs}", flush=True)
        del branch

    print("\n2. Treniranje pre završne provere", flush=True)

    branches = fit_ensemble(
        sequences[:test_start],
        cdf[:test_start],
        epochs,
    )

    pattern_model = PatternModel().fit(
        sequences[:test_start],
        patterns[:test_start],
        targets[:test_start],
    )

    test_pmf = predict_ensemble(
        branches, sequences[test_start:count]
    )

    predictions = []
    hit_counts = []
    control_hits = []

    for index in range(TEST):
        target_index = test_start + index
        previous = draws[SEQUENCE + target_index - 1]

        combination, _, evaluated = select_next(
            test_pmf[index],
            pattern_model,
            sequences[target_index],
            previous,
        )

        actual = targets[target_index]

        predictions.append(combination)
        hit_counts.append(
            len(set(combination).intersection(actual))
        )
        control_hits.append(
            len(set(previous).intersection(actual))
        )

        if (index + 1) % 8 == 0:
            print(
                f"  Provera {index + 1}/{TEST}; "
                f"ocenjenih kandidata poslednjeg kola: {evaluated}",
                flush=True,
            )

    hit_counts = np.asarray(hit_counts)
    control_hits = np.asarray(control_hits)

    previous_rows = draws[SEQUENCE + test_start - 1:-1]
    copied_previous = sum(
        tuple(predicted) == tuple(previous)
        for predicted, previous in zip(predictions, previous_rows)
    )

    print("\nODVOJENA PROVERA — POSLEDNJA 64 KOLA")
    print(f"Prosečno pogodaka: {hit_counts.mean():.4f}")
    print(
        "Raspodela pogodaka 0..7: "
        f"{np.bincount(hit_counts, minlength=8).tolist()}"
    )
    print(f"Kola sa 3+: {int(np.sum(hit_counts >= 3))}")
    print(f"Kola sa 4+: {int(np.sum(hit_counts >= 4))}")
    print(
        "Kontrola: prethodno izvuceno kolo — "
        f"{control_hits.mean():.4f}"
    )
    print(
        "Razlika prema kontroli: "
        f"{hit_counts.mean() - control_hits.mean():+.4f}"
    )
    print(f"Različite prognoze: {len(set(predictions))}/{TEST}")
    print(f"Kopije prethodnog kola: {copied_previous}")
    print(
        "Prognoze sa 5+ uzastopnih: "
        f"{sum(longest_run(item) >= 5 for item in predictions)}"
    )
    print(
        "Prognoze potpuno u jednoj polovini: "
        f"{sum(max(item) <= 19 or min(item) >= 20 for item in predictions)}"
    )

    del branches, pattern_model, test_pmf

    print("\n3. Finalno treniranje na svim prelazima", flush=True)

    branches = fit_ensemble(
        sequences[:-1], cdf, epochs
    )

    pattern_model = PatternModel().fit(
        sequences[:-1], patterns, targets
    )

    final_pmf = predict_ensemble(
        branches, sequences[-1:]
    )[0]

    combination, score, evaluated = select_next(
        final_pmf,
        pattern_model,
        sequences[-1],
        draws[-1],
    )

    print("\nFINALNI NEXT: " + ", ".join(map(str, combination)))
    print(f"Modelski skor za rangiranje: {score:.8f}")
    print(f"Ocenjenih različitih kandidata: {evaluated}")
    print(f"Najduži uzastopni niz: {longest_run(combination)}")

    lower = sum(number <= 19 for number in combination)
    print(f"Brojeva 1..19 / 20..39: {lower} / {7 - lower}")
    print("Neuronski ponderi: Tabular=0.25, LSTM=0.35, Transformer=0.40")
    print("Pretraga je deterministička i heuristička.")

    return combination


def main():
    parser = argparse.ArgumentParser(
        description="DLT arhitektura prilagođena Loto 7/39 CSV-ovima."
    )
    parser.add_argument("csv", nargs="*")
    args = parser.parse_args()

    filenames = args.csv or DEFAULT_FILES

    if len(filenames) != 3:
        raise ValueError("Navedi tačno tri CSV putanje.")

    datasets = [load_csv(filename) for filename in filenames]

    if len({path for path, _, _ in datasets}) != 3:
        raise ValueError("Potrebna su tri različita CSV fajla.")

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    print("DLT LOTO V1")
    print("Tabularna pažnja + BiLSTM + Transformer.")
    print("Više bazena kandidata + distribucija obrazaca + evolucija.")
    print("Svaki CSV: zasebno treniranje i jedna završna kombinacija.")
    print("Cela istorija za finalno treniranje.")
    print("Bez random inicijalizacije, dropout-a i mešanja uzoraka.")
    print("Redovi moraju biti od najstarijeg ka najnovijem.")
    print("Napredak treniranja i provere ispisuje se tokom rada.")

    results = []

    for path, draws, fingerprint in datasets:
        combination = process_dataset(path, draws, fingerprint)
        results.append((path.name, combination))

    print("\n" + "=" * 76)
    print("ZAVRŠNI NEXT PREGLED")

    for name, combination in results:
        print(f"{name}: " + ", ".join(map(str, combination)))

    grouped = {}

    for name, combination in results:
        grouped.setdefault(combination, []).append(name)

    for combination, names in grouped.items():
        if len(names) > 1:
            print(
                "\nIsti NEXT: "
                + " / ".join(names)
                + ": "
                + ", ".join(map(str, combination))
            )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nPrekinuto.", file=sys.stderr)
        sys.exit(130)
    except (
        OSError,
        ValueError,
        RuntimeError,
        FloatingPointError,
        csv.Error,
    ) as exc:
        print(f"\nGREŠKA: {exc}", file=sys.stderr)
        sys.exit(1)



"""
DLT LOTO V1
Tabularna pažnja + BiLSTM + Transformer.
Više bazena kandidata + distribucija obrazaca + evolucija.
Svaki CSV: zasebno treniranje i jedna završna kombinacija.
Cela istorija za finalno treniranje.
Bez random inicijalizacije, dropout-a i mešanja uzoraka.
Redovi moraju biti od najstarijeg ka najnovijem.
Napredak treniranja i provere ispisuje se tokom rada.


============================================================================
loto7_4704_k83.csv
Izvlačenja: 4704
Prelaza za finalno treniranje: 4684
SHA-256: 1decd748cc0c398e384d9b16fd7e7eb50a5396f59a2f2cf9e7cdd05bdd07ae09
Validacija — CSV redovi: 4593..4640
Završna provera — CSV redovi: 4641..4704

1. Hronološki izbor dužine treniranja
  Validacija: Tabular_attention
    Tabular_attention: epoha 5/30
    Tabular_attention: epoha 10/30
    Tabular_attention: epoha 15/30
    Izabrano epoha: 11
  Validacija: BiLSTM
    BiLSTM: epoha 5/30
    BiLSTM: epoha 10/30
    BiLSTM: epoha 15/30
    BiLSTM: epoha 20/30
    BiLSTM: epoha 25/30
    Izabrano epoha: 21
  Validacija: Transformer
    Transformer: epoha 5/30
    Transformer: epoha 10/30
    Transformer: epoha 15/30
    Transformer: epoha 20/30
    Izabrano epoha: 18

2. Treniranje pre završne provere
  Treniranje: Tabular_attention
    Tabular_attention: epoha 5/11
    Tabular_attention: epoha 10/11
  Treniranje: BiLSTM
    BiLSTM: epoha 5/21
    BiLSTM: epoha 10/21
    BiLSTM: epoha 15/21
    BiLSTM: epoha 20/21
  Treniranje: Transformer
    Transformer: epoha 5/18
    Transformer: epoha 10/18
    Transformer: epoha 15/18
  Provera 8/64; ocenjenih kandidata poslednjeg kola: 1706
  Provera 16/64; ocenjenih kandidata poslednjeg kola: 2789
  Provera 24/64; ocenjenih kandidata poslednjeg kola: 2954
  Provera 32/64; ocenjenih kandidata poslednjeg kola: 3326
  Provera 40/64; ocenjenih kandidata poslednjeg kola: 3208
  Provera 48/64; ocenjenih kandidata poslednjeg kola: 2769
  Provera 56/64; ocenjenih kandidata poslednjeg kola: 1951
  Provera 64/64; ocenjenih kandidata poslednjeg kola: 1700

ODVOJENA PROVERA — POSLEDNJA 64 KOLA
Prosečno pogodaka: 1.3125
Raspodela pogodaka 0..7: [14, 25, 16, 9, 0, 0, 0, 0]
Kola sa 3+: 9
Kola sa 4+: 0
Kontrola: prethodno izvuceno kolo — 1.1406
Razlika prema kontroli: +0.1719
Različite prognoze: 48/64
Kopije prethodnog kola: 0
Prognoze sa 5+ uzastopnih: 0
Prognoze potpuno u jednoj polovini: 0

3. Finalno treniranje na svim prelazima
  Treniranje: Tabular_attention
    Tabular_attention: epoha 5/11
    Tabular_attention: epoha 10/11
  Treniranje: BiLSTM
    BiLSTM: epoha 5/21
    BiLSTM: epoha 10/21
    BiLSTM: epoha 15/21
    BiLSTM: epoha 20/21
  Treniranje: Transformer
    Transformer: epoha 5/18
    Transformer: epoha 10/18
    Transformer: epoha 15/18

FINALNI NEXT: 4, x, 18, y, 30, z, 35
Modelski skor za rangiranje: -1.60017151
Ocenjenih različitih kandidata: 2480
Najduži uzastopni niz: 2
Brojeva 1..19 / 20..39: 3 / 4
Neuronski ponderi: Tabular=0.25, LSTM=0.35, Transformer=0.40
Pretraga je deterministička i heuristička.



============================================================================
loto7_4704_k83_loto_2974.csv
Izvlačenja: 2974
Prelaza za finalno treniranje: 2954
SHA-256: fc5313380978e4826789413029140d99f09bd92c143892bf9b9027ff9f276be7
Validacija — CSV redovi: 2863..2910
Završna provera — CSV redovi: 2911..2974

1. Hronološki izbor dužine treniranja
  Validacija: Tabular_attention
    Tabular_attention: epoha 5/30
    Tabular_attention: epoha 10/30
    Tabular_attention: epoha 15/30
    Izabrano epoha: 10
  Validacija: BiLSTM
    BiLSTM: epoha 5/30
    BiLSTM: epoha 10/30
    BiLSTM: epoha 15/30
    BiLSTM: epoha 20/30
    BiLSTM: epoha 25/30
    Izabrano epoha: 24
  Validacija: Transformer
    Transformer: epoha 5/30
    Transformer: epoha 10/30
    Transformer: epoha 15/30
    Transformer: epoha 20/30
    Izabrano epoha: 17

2. Treniranje pre završne provere
  Treniranje: Tabular_attention
    Tabular_attention: epoha 5/10
    Tabular_attention: epoha 10/10
  Treniranje: BiLSTM
    BiLSTM: epoha 5/24
    BiLSTM: epoha 10/24
    BiLSTM: epoha 15/24
    BiLSTM: epoha 20/24
  Treniranje: Transformer
    Transformer: epoha 5/17
    Transformer: epoha 10/17
    Transformer: epoha 15/17
  Provera 8/64; ocenjenih kandidata poslednjeg kola: 2398
  Provera 16/64; ocenjenih kandidata poslednjeg kola: 2733
  Provera 24/64; ocenjenih kandidata poslednjeg kola: 2132
  Provera 32/64; ocenjenih kandidata poslednjeg kola: 3059
  Provera 40/64; ocenjenih kandidata poslednjeg kola: 2177
  Provera 48/64; ocenjenih kandidata poslednjeg kola: 1942
  Provera 56/64; ocenjenih kandidata poslednjeg kola: 2149
  Provera 64/64; ocenjenih kandidata poslednjeg kola: 2506

ODVOJENA PROVERA — POSLEDNJA 64 KOLA
Prosečno pogodaka: 1.5000
Raspodela pogodaka 0..7: [9, 25, 19, 11, 0, 0, 0, 0]
Kola sa 3+: 11
Kola sa 4+: 0
Kontrola: prethodno izvuceno kolo — 1.3125
Razlika prema kontroli: +0.1875
Različite prognoze: 38/64
Kopije prethodnog kola: 0
Prognoze sa 5+ uzastopnih: 0
Prognoze potpuno u jednoj polovini: 0

3. Finalno treniranje na svim prelazima
  Treniranje: Tabular_attention
    Tabular_attention: epoha 5/10
    Tabular_attention: epoha 10/10
  Treniranje: BiLSTM
    BiLSTM: epoha 5/24
    BiLSTM: epoha 10/24
    BiLSTM: epoha 15/24
    BiLSTM: epoha 20/24
  Treniranje: Transformer
    Transformer: epoha 5/17
    Transformer: epoha 10/17
    Transformer: epoha 15/17

FINALNI NEXT: 4, x, 16, y, 23, z, 35
Modelski skor za rangiranje: -1.58254398
Ocenjenih različitih kandidata: 2222
Najduži uzastopni niz: 2
Brojeva 1..19 / 20..39: 4 / 3
Neuronski ponderi: Tabular=0.25, LSTM=0.35, Transformer=0.40
Pretraga je deterministička i heuristička.



============================================================================
loto7_4704_k83_loto_plus_1730.csv
Izvlačenja: 1730
Prelaza za finalno treniranje: 1710
SHA-256: 1f4252375372cbbb88cd026892dc19eb5400e6386bfc5dc613ff7d19d9db0504
Validacija — CSV redovi: 1619..1666
Završna provera — CSV redovi: 1667..1730

1. Hronološki izbor dužine treniranja
  Validacija: Tabular_attention
    Tabular_attention: epoha 5/30
    Tabular_attention: epoha 10/30
    Tabular_attention: epoha 15/30
    Tabular_attention: epoha 20/30
    Tabular_attention: epoha 25/30
    Tabular_attention: epoha 30/30
    Izabrano epoha: 27
  Validacija: BiLSTM
    BiLSTM: epoha 5/30
    BiLSTM: epoha 10/30
    BiLSTM: epoha 15/30
    BiLSTM: epoha 20/30
    BiLSTM: epoha 25/30
    BiLSTM: epoha 30/30
    Izabrano epoha: 30
  Validacija: Transformer
    Transformer: epoha 5/30
    Transformer: epoha 10/30
    Transformer: epoha 15/30
    Transformer: epoha 20/30
    Transformer: epoha 25/30
    Transformer: epoha 30/30
    Izabrano epoha: 29

2. Treniranje pre završne provere
  Treniranje: Tabular_attention
    Tabular_attention: epoha 5/27
    Tabular_attention: epoha 10/27
    Tabular_attention: epoha 15/27
    Tabular_attention: epoha 20/27
    Tabular_attention: epoha 25/27
  Treniranje: BiLSTM
    BiLSTM: epoha 5/30
    BiLSTM: epoha 10/30
    BiLSTM: epoha 15/30
    BiLSTM: epoha 20/30
    BiLSTM: epoha 25/30
    BiLSTM: epoha 30/30
  Treniranje: Transformer
    Transformer: epoha 5/29
    Transformer: epoha 10/29
    Transformer: epoha 15/29
    Transformer: epoha 20/29
    Transformer: epoha 25/29
  Provera 8/64; ocenjenih kandidata poslednjeg kola: 2327
  Provera 16/64; ocenjenih kandidata poslednjeg kola: 2774
  Provera 24/64; ocenjenih kandidata poslednjeg kola: 3195
  Provera 32/64; ocenjenih kandidata poslednjeg kola: 2133
  Provera 40/64; ocenjenih kandidata poslednjeg kola: 2572
  Provera 48/64; ocenjenih kandidata poslednjeg kola: 2585
  Provera 56/64; ocenjenih kandidata poslednjeg kola: 1946
  Provera 64/64; ocenjenih kandidata poslednjeg kola: 1874

ODVOJENA PROVERA — POSLEDNJA 64 KOLA
Prosečno pogodaka: 1.3906
Raspodela pogodaka 0..7: [10, 28, 18, 7, 1, 0, 0, 0]
Kola sa 3+: 8
Kola sa 4+: 1
Kontrola: prethodno izvuceno kolo — 1.2812
Razlika prema kontroli: +0.1094
Različite prognoze: 47/64
Kopije prethodnog kola: 0
Prognoze sa 5+ uzastopnih: 0
Prognoze potpuno u jednoj polovini: 0

3. Finalno treniranje na svim prelazima
  Treniranje: Tabular_attention
    Tabular_attention: epoha 5/27
    Tabular_attention: epoha 10/27
    Tabular_attention: epoha 15/27
    Tabular_attention: epoha 20/27
    Tabular_attention: epoha 25/27
  Treniranje: BiLSTM
    BiLSTM: epoha 5/30
    BiLSTM: epoha 10/30
    BiLSTM: epoha 15/30
    BiLSTM: epoha 20/30
    BiLSTM: epoha 25/30
    BiLSTM: epoha 30/30
  Treniranje: Transformer
    Transformer: epoha 5/29
    Transformer: epoha 10/29
    Transformer: epoha 15/29
    Transformer: epoha 20/29
    Transformer: epoha 25/29

FINALNI NEXT: 1, x, 11, y, 23, z, 35
Modelski skor za rangiranje: -1.59879089
Ocenjenih različitih kandidata: 2248
Najduži uzastopni niz: 2
Brojeva 1..19 / 20..39: 4 / 3
Neuronski ponderi: Tabular=0.25, LSTM=0.35, Transformer=0.40
Pretraga je deterministička i heuristička.


============================================================================
ZAVRŠNI NEXT PREGLED
loto7_4704_k83.csv: 4, x, 18, y, 30, z, 35
loto7_4704_k83_loto_2974.csv: 4, x, 16, y, 23, z, 35
loto7_4704_k83_loto_plus_1730.csv: 1, x, 11, y, 23, z, 35
"""



"""
Spajanje više strategija za stvaranje i rangiranje kandidata, uz neuronski ansambl: 
LSTM, Transformer i pojednostavljena mreža sa pažnjom nazvana TabNet.
Ovde se spajaju ocene više strategija.

- Jedna kombinacija 7/39, zasebno za svaki ceo CSV.
- Distribucije.
- Neuronske arhitekture, a izlazi i gubitak prilagodjeni regresiji.
- Deterministička pretraga.
- Kvalitet kombinacije treba da određuju naučene ocene.
- Hronološka validaciona podela.


Mreža sa pažnjom, dvosmerni LSTM, Transformer, spajanje strategija i evolutivno poboljšavanje kandidata. 
Ponderi neuronskog ansambla su 25/35/40. 
Bazeni kandidata prilagođeni su naučenim distribucijama, a izbor, ukrštanje i mutacije su deterministički.
"""
