# Copyright (c) 2025 The Regents of the University of California
#
# This file is part of BRAILS++.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# You should have received a copy of the BSD 3-Clause License along with
# BRAILS++. If not, see <http://www.opensource.org/licenses/>.

"""
Derive reproducible random draws from asset identity, not from order.

A draw that must reproduce bit for bit under reordering, subsetting, and
reruns cannot come from a stream generator, because a generator ties the
value an asset receives to its position in the sequence. These helpers
derive one uniform per (asset, world, purpose, iteration) from a keyed
hash instead, so an asset draws the same number whether it runs alone
or as part of a large inventory.

.. autosummary::

      asset_key_digest
      keyed_uniforms
      truncated_normal
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import numpy as np
from scipy.special import ndtr, ndtri

if TYPE_CHECKING:
    from collections.abc import Iterable

# SplitMix64, used as a finalizer: the scrambling is what is wanted here,
# not the sequence. The algorithm and its golden-ratio additive constant are
# Steele, Lea, and Flood (2014); the multipliers and shift widths are the
# variant of the MurmurHash3 finalizer that the reference splitmix64
# implementation carries. Taken here for their avalanche behavior: every
# input bit reaches every output bit.
_GOLDEN_GAMMA = np.uint64(0x9E3779B97F4A7C15)
_MULTIPLIER_A = np.uint64(0xBF58476D1CE4E5B9)
_MULTIPLIER_B = np.uint64(0x94D049BB133111EB)
_SHIFT_A = np.uint64(30)
_SHIFT_B = np.uint64(27)
_SHIFT_C = np.uint64(31)

# A uniform is the top 53 bits of the hash scaled into [0, 1), 53 being
# the number of bits a float64 mantissa holds exactly.
_MANTISSA_SHIFT = np.uint64(11)
_MANTISSA_SCALE = 2.0**-53

_MASK_64 = 0xFFFFFFFFFFFFFFFF
_DIGEST_BYTES = 8

# ndtri maps 0 and 1 to infinities, and a uniform of exactly 0 is drawn
# once in 2**53. Nudging the probability inside the open interval keeps
# the inverse finite; the caller's clip puts it back on the bounds.
_PROBABILITY_FLOOR = float(np.nextafter(0.0, 1.0))
_PROBABILITY_CEILING = float(np.nextafter(1.0, 0.0))

DEFAULT_WORLD = 0
DEFAULT_ITERATION = 0


def _splitmix64(value: np.ndarray) -> np.ndarray:
    """
    Avalanche a 64-bit value.

    Args:
        value (np.ndarray):
            Unsigned 64-bit values to mix.

    Returns:
        np.ndarray:
            The mixed values, same shape and dtype.
    """
    with np.errstate(over='ignore'):
        mixed = value + _GOLDEN_GAMMA
        mixed = (mixed ^ (mixed >> _SHIFT_A)) * _MULTIPLIER_A
        mixed = (mixed ^ (mixed >> _SHIFT_B)) * _MULTIPLIER_B
    return mixed ^ (mixed >> _SHIFT_C)


def _digest(text: str) -> np.uint64:
    """
    Hash a string into a 64-bit key.

    Args:
        text (str):
            The string to hash.

    Returns:
        np.uint64:
            The first eight bytes of its BLAKE2b digest.
    """
    raw = hashlib.blake2b(
        text.encode('utf-8'), digest_size=_DIGEST_BYTES
    ).digest()
    return np.uint64(int.from_bytes(raw, 'little'))


def asset_key_digest(asset_ids: Iterable[str | int]) -> np.ndarray:
    """
    Turn asset ids into stable 64-bit keys.

    Ids are canonicalized to strings before hashing, so an inventory keeps
    its draws when integer ids return as strings from a GeoJSON round
    trip. Digest once and reuse the result: hashing dominates the cost
    of a draw, and callers commonly redraw the same assets many times.

    Args:
        asset_ids (Iterable[str | int]):
            The ids to digest, in any order.

    Returns:
        np.ndarray:
            Unsigned 64-bit keys, one per id, in the order given.
    """
    return np.fromiter(
        (_digest(str(asset_id)) for asset_id in asset_ids), dtype=np.uint64
    )


def keyed_uniforms(
    asset_keys: np.ndarray,
    *,
    seed: int,
    purpose: str,
    world: int = DEFAULT_WORLD,
    iteration: int = DEFAULT_ITERATION,
) -> np.ndarray:
    """
    Draw one uniform per asset from its key.

    The seed, key, world, purpose, and iteration are folded through the
    mixer one at a time. Combining them arithmetically instead — a seed
    plus an iteration counter, say — would leave neighboring draws
    related, which matters when a caller asks the same asset for a
    fresh value on one attempt after another.

    Args:
        asset_keys (np.ndarray):
            Keys from :func:`asset_key_digest`.
        seed (int):
            The user's seed for the compilation.
        purpose (str):
            What the uniform is for. Purpose labels are part of the
            reproducibility contract: renaming one changes every draw it
            governs, so treat them as constants, not as free text.
        world (int):
            Possible-world index.
        iteration (int):
            Resolution-loop attempt number.

    Returns:
        np.ndarray:
            Uniforms in [0, 1), one per key, in the order given.
    """
    keys = np.asarray(asset_keys, dtype=np.uint64)
    mixed = _splitmix64(np.full(keys.shape, np.uint64(seed & _MASK_64)))
    mixed = _splitmix64(mixed ^ keys)
    mixed = _splitmix64(mixed ^ np.uint64(world & _MASK_64))
    mixed = _splitmix64(mixed ^ _digest(purpose))
    mixed = _splitmix64(mixed ^ np.uint64(iteration & _MASK_64))
    return (mixed >> _MANTISSA_SHIFT).astype(np.float64) * _MANTISSA_SCALE


def truncated_normal(
    uniforms: np.ndarray,
    *,
    mean: np.ndarray | float,
    sigma: float,
    lower: np.ndarray | float,
    upper: np.ndarray | float,
) -> np.ndarray:
    """
    Invert uniforms through a truncated normal.

    The uniform is mapped into the CDF interval the bounds carve out and
    inverted there.

    Args:
        uniforms (np.ndarray):
            Uniforms in [0, 1).
        mean (np.ndarray | float):
            Distribution mean, per asset or shared.
        sigma (float):
            Standard deviation before truncation.
        lower (np.ndarray | float):
            Lower bound, per asset or shared.
        upper (np.ndarray | float):
            Upper bound, per asset or shared.

    Returns:
        np.ndarray:
            Draws inside [lower, upper].

    Raises:
        ValueError:
            If sigma is not positive, if a bound pair is inverted, or if
            an interval sits so far into a tail that it carries no
            representable probability.
    """
    if sigma <= 0:
        raise ValueError(f'sigma must be positive, got {sigma!r}.')
    lower_bound = np.asarray(lower, dtype=np.float64)
    upper_bound = np.asarray(upper, dtype=np.float64)
    if np.any(upper_bound <= lower_bound):
        raise ValueError(
            'Every upper bound must sit above its lower bound.'
        )
    center = np.asarray(mean, dtype=np.float64)
    lower_cdf = ndtr((lower_bound - center) / sigma)
    upper_cdf = ndtr((upper_bound - center) / sigma)
    if np.any(upper_cdf <= lower_cdf):
        raise ValueError(
            'A truncation interval carries no representable probability: '
            'its bounds sit too far into the tail of the distribution.'
        )
    probability = lower_cdf + uniforms * (upper_cdf - lower_cdf)
    probability = np.clip(
        probability, _PROBABILITY_FLOOR, _PROBABILITY_CEILING
    )
    return np.clip(
        center + sigma * ndtri(probability), lower_bound, upper_bound
    )
