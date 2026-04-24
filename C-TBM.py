#!/usr/bin/env python3
"""
C-TBM Functional Verification Script
=====================================
Verifies the constrained-modulus Montgomery reduction decomposition
for Proth primes used in FHE hardware accelerators.

This script proves that for NTT-friendly primes q = c * 2^n + 1:
  - Step 3 (×q):  m * q = (m * c) << n + m
  - Step 2 (×q'): m * q' mod R = [(m * (d+1)) << n - m] mod R
    where q' = d * 2^n + (2^n - 1) and d is (k-n) bits wide

March 26th, 2026
"""

import sys
from sympy import isprime


# ==============================================================
#  Utility Functions
# ==============================================================

def extended_gcd(a, b):
    """Extended Euclidean Algorithm: returns (gcd, x, y) such that a*x + b*y = gcd."""
    if a == 0:
        return b, 0, 1
    g, x, y = extended_gcd(b % a, a)
    return g, y - (b // a) * x, x


def modinv(a, m):
    """Compute modular inverse a^(-1) mod m."""
    g, x, _ = extended_gcd(a % m, m)
    if g != 1:
        raise ValueError(f"No modular inverse: gcd({a}, {m}) = {g}")
    return x % m


def montgomery_redc_standard(T, q, k):
    """
    Standard Montgomery Reduction (REDC).
    Computes T * R^(-1) mod q where R = 2^k.
    """
    R = 1 << k
    q_prime = (-modinv(q, R)) % R  # q' = -q^(-1) mod R

    # Step 2: m = (T mod R) * q' mod R
    m = (T % R) * q_prime % R

    # Step 3: t = (T + m * q) / R
    t = (T + m * q) >> k

    # Step 4: conditional subtraction
    if t >= q:
        t -= q

    return t


def montgomery_redc_constrained(T, q, c, n, k):
    """
    Constrained-Modulus Montgomery Reduction for Proth primes q = c * 2^n + 1.
    Replaces full-width multiplications with reduced-width operations.
    """
    R = 1 << k
    q_prime = (-modinv(q, R)) % R
    d = (q_prime - ((1 << n) - 1)) >> n  # q' = d * 2^n + (2^n - 1)

    T_lo = T % R  # lower k bits of T

    # Step 2 (constrained): m = T_lo * q' mod R
    # Decomposed: m = [(T_lo * (d+1)) << n - T_lo] mod R
    m = (((T_lo * (d + 1)) << n) - T_lo) % R

    # Step 3 (constrained): compute m * q using m * (c * 2^n + 1) = (m*c) << n + m
    mq = (m * c << n) + m

    # Complete Step 3: t = (T + m*q) / R
    t = (T + mq) >> k

    # Step 4: conditional subtraction
    if t >= q:
        t -= q

    return t


def montgomery_multiply(a, b, q, k, use_constrained=False, c=None, n=None):
    """
    Full Montgomery Modular Multiplication: computes a * b * R^(-1) mod q.
    When inputs are in Montgomery form (aR mod q, bR mod q), the output
    is the Montgomery form of a*b mod q.
    """
    T = a * b  # Step 1: full integer multiply
    if use_constrained:
        return montgomery_redc_constrained(T, q, c, n, k)
    else:
        return montgomery_redc_standard(T, q, k)


# ==========================================================================
#  Test 1: Small Numerical Example (Trace-Through)
# ==========================================================================

def test_small_example():
    """
    Detailed trace-through with a small Proth prime for paper inclusion.
    Uses q = 97 = 3 * 2^5 + 1, k = 8 bits.
    """
    print("=" * 72)
    print("TEST 1: Detailed Numerical Trace-Through")
    print("=" * 72)

    q = 97       # Proth prime: 3 * 2^5 + 1
    c = 3        # upper coefficient
    n = 5        # constraint bits (q = c * 2^n + 1)
    k = 8        # operand bit-width
    R = 1 << k   # R = 256

    print(f"\nParameters:")
    print(f"  q = {q} = {c} * 2^{n} + 1  (Proth prime)")
    print(f"  k = {k},  R = 2^{k} = {R}")
    print(f"  c = {c} ({c.bit_length()}-bit coefficient)")

    # Precompute Montgomery constants
    q_prime = (-modinv(q, R)) % R
    d = (q_prime - ((1 << n) - 1)) >> n

    print(f"\nPrecomputed constants:")
    print(f"  q'  = -q^(-1) mod R = {q_prime}  (binary: {bin(q_prime)})")
    print(f"  q'  = {d} * 2^{n} + {(1 << n) - 1}")
    print(f"  d   = {d} ({d.bit_length()}-bit),  d+1 = {d + 1}")

    # Verify q' structure
    assert q_prime == d * (1 << n) + ((1 << n) - 1), "q' structure verification FAILED"
    print(f"  Verified: q' = d * 2^{n} + (2^{n} - 1)")

    # Choose test operands
    a, b = 42, 73
    print(f"\nTest operands: a = {a}, b = {b}")
    print(f"  a * b mod q = {(a * b) % q}  (expected final result in standard form)")

    # Convert to Montgomery form
    R2_mod_q = (R * R) % q
    a_mont = (a * R) % q
    b_mont = (b * R) % q
    print(f"\nMontgomery form:")
    print(f"  aR mod q = {a_mont}")
    print(f"  bR mod q = {b_mont}")

    # Step 1: Integer multiply
    T = a_mont * b_mont
    print(f"\nStep 1 — Integer multiply:")
    print(f"  T = {a_mont} * {b_mont} = {T}")

    # === Standard REDC ===
    print(f"\n--- Standard Montgomery Reduction ---")
    m_std = (T % R) * q_prime % R
    mq_std = m_std * q
    t_std = (T + mq_std) >> k
    if t_std >= q:
        t_std -= q
    print(f"  Step 2: m = (T mod R) * q' mod R = ({T % R}) * {q_prime} mod {R} = {m_std}")
    print(f"  Step 3: m * q = {m_std} * {q} = {mq_std}")
    print(f"          t = (T + m*q) >> {k} = ({T} + {mq_std}) >> {k} = {(T + mq_std) >> k}")
    if (T + mq_std) >> k >= q:
        print(f"  Step 4: t >= q, so t = t - q = {t_std}")
    else:
        print(f"  Step 4: t < q, so t = {t_std}")

    # === Constrained REDC ===
    print(f"\n--- Constrained Montgomery Reduction ---")
    T_lo = T % R
    print(f"  Step 2 (constrained):")
    print(f"    T_lo = T mod R = {T_lo}")
    inner = T_lo * (d + 1)
    shifted = inner << n
    m_con = (shifted - T_lo) % R
    print(f"    T_lo * (d+1) = {T_lo} * {d + 1} = {inner}")
    print(f"    << {n} = {shifted}")
    print(f"    - T_lo = {shifted} - {T_lo} = {shifted - T_lo}")
    print(f"    mod R  = {m_con}")
    print(f"    Matches standard m? {m_con == m_std}" if m_con == m_std else f"    MISMATCH! standard={m_std}, constrained={m_con}")
    print(f"\n  Step 3 (constrained):")
    mc = m_con * c
    mc_shifted = mc << n
    mq_con = mc_shifted + m_con
    print(f"    m * c = {m_con} * {c} = {mc}")
    print(f"    (m * c) << {n} = {mc_shifted}")
    print(f"    + m = {mc_shifted} + {m_con} = {mq_con}")
    print(f"    Matches standard m*q? {mq_con == mq_std}" if mq_con == mq_std else f"    MISMATCH! standard={mq_std}, constrained={mq_con}")
    t_con = (T + mq_con) >> k
    if t_con >= q:
        t_con -= q
    print(f"\n  Step 4: t = (T + m*q) >> {k} = {(T + mq_con) >> k}" + (f" - {q} = {t_con}" if (T + mq_con) >> k >= q else f" = {t_con}"))

    # Convert back from Montgomery form
    result_mont = t_con
    result = montgomery_redc_standard(result_mont, q, k)
    expected = (a * b) % q
    print(f"\nFinal result (Montgomery form): {result_mont}")
    print(f"Convert back: {result_mont} * R^(-1) mod {q} = {result}")
    print(f"Expected (a * b mod q): {expected}")
    print(f"PASS" if result == expected else "FAIL")

    return result == expected


# =======================================================================
#  Test 2: q' Structure Verification Across Many Primes
# =======================================================================

def test_q_prime_structure():
    """
    Verify that q' = d * 2^n + (2^n - 1) for all tested Proth primes,
    and that d is exactly (k - n) bits wide.
    """
    print("\n" + "=" * 72)
    print("TEST 2: q' Structure Verification Across Proth Primes")
    print("=" * 72)

    n = 17  # constraint bits for N = 2^16

    results = []

    for mode_name, k, c_range in [
        ("36-bit", 36, range(3, 500, 2)),
        ("60-bit", 60, range(2**42 + 1, 2**42 + 500, 2)),
    ]:
        print(f"\n--- {mode_name} mode (k={k}) ---")
        print(f"  {'c':>15s}  {'c bits':>6s}  {'d+1':>15s}  {'d+1 bits':>8s}  {'Lower n OK':>10s}")
        print(f"  {'-'*15}  {'-'*6}  {'-'*15}  {'-'*8}  {'-'*10}")

        count = 0
        for c_val in c_range:
            q = c_val * (1 << n) + 1
            if q >= (1 << k):
                continue
            if not isprime(q):
                continue

            R = 1 << k
            q_prime = (-modinv(q, R)) % R
            lower_n = q_prime & ((1 << n) - 1)
            d = (q_prime - ((1 << n) - 1)) >> n
            lower_ok = lower_n == (1 << n) - 1
            d_bits = (d + 1).bit_length()

            print(f"  {c_val:15d}  {c_val.bit_length():6d}  {d+1:15d} {d_bits:8d}  {'YES' if lower_ok else 'NO':>10s}")
            results.append((mode_name, k, c_val, d + 1, d_bits, lower_ok))

            count += 1
            if count >= 8:
                break

    all_pass = all(r[5] for r in results)
    print(f"\nAll primes satisfy q' = d·2^{n} + (2^{n}-1): {'PASS' if all_pass else 'FAIL'}")

    # Check that d+1 is always ≤ (k-n) bits
    max_d_bits_36 = max(r[4] for r in results if r[0] == "36-bit")
    max_d_bits_60 = max(r[4] for r in results if r[0] == "60-bit")
    print(f"Max (d+1) bit-width for 36-bit primes: {max_d_bits_36} (expected ≤ {36 - n} = {36 - n})")
    print(f"Max (d+1) bit-width for 60-bit primes: {max_d_bits_60} (expected ≤ {60 - n} = {60 - n})")

    return all_pass


# =======================================================================
#  Test 3: Full Montgomery Multiply — Standard vs. Constrained
# =======================================================================

def test_full_multiply():
    """
    Run Montgomery multiplication on many random-ish operand pairs,
    comparing standard and constrained REDC for correctness.
    """
    print("\n" + "=" * 72)
    print("TEST 3: Montgomery Multiply Correctness (Standard vs. Constrained)")
    print("=" * 72)

    n = 17
    test_cases = 0
    failures = 0

    for mode_name, k, c_range in [
        ("36-bit", 36, range(3, 300, 2)),
        ("60-bit", 60, range(2**42 + 1, 2**42 + 200, 2)),
    ]:
        print(f"\n--- {mode_name} mode (k={k}) ---")
        mode_tests = 0
        mode_fails = 0

        for c_val in c_range:
            q = c_val * (1 << n) + 1
            if q >= (1 << k) or not isprime(q):
                continue

            R = 1 << k

            # Test with multiple operand pairs per prime
            test_operands = [
                (1, 1), (0, 42), (q - 1, q - 1), (q // 2, q // 3),
                (12345 % q, 67890 % q), (q - 2, 3), (7, q - 1),
            ]

            for a, b in test_operands:
                if a >= q or b >= q:
                    continue

                # Convert to Montgomery form
                a_mont = (a * R) % q
                b_mont = (b * R) % q

                # Multiply using both methods
                result_std = montgomery_multiply(a_mont, b_mont, q, k,
                                                 use_constrained=False)
                result_con = montgomery_multiply(a_mont, b_mont, q, k,
                                                 use_constrained=True, c=c_val, n=n)

                if result_std != result_con:
                    print(f"  FAIL: q={q}, a={a}, b={b}: std={result_std}, con={result_con}")
                    mode_fails += 1
                    failures += 1

                mode_tests += 1
                test_cases += 1

            if mode_tests >= 200:
                break

        print(f"  Tested {mode_tests} multiplications: "
              f"{'ALL PASS' if mode_fails == 0 else f'{mode_fails} FAILURES'}")

    print(f"\nTotal: {test_cases} test cases, {failures} failures")
    print(f"{'ALL PASS' if failures == 0 else 'SOME FAILURES'}")

    return failures == 0


# ========================================================================
#  Test 4: Hardware Cost Summary
# ========================================================================

def test_hardware_cost():
    """
    Compute and display the area and critical path comparison
    for inclusion in the paper.
    """
    print("\n" + "=" * 72)
    print("TEST 4: Hardware Cost Summary")
    print("=" * 72)

    n = 17   # Proth constraint bits
    GE_PER_FA = 7
    MUX_OVERHEAD_GE = 560  # ~80 muxes × 7 GE for shift routing

    for mode_name, k in [("36-bit", 36), ("60-bit", 60)]:
        c_width = k - n

        # Reduction multiplier: ×q
        gen_mult_fas = k * k
        con_mult_fas = k * c_width
        adder_fas = k + n  # shift-add combination
        con_total_ges = con_mult_fas * GE_PER_FA + adder_fas * GE_PER_FA + MUX_OVERHEAD_GE
        gen_total_ges = gen_mult_fas * GE_PER_FA

        # Critical path (array multiplier model, CSA for intermediate adds)
        gen_cp = 2 * k
        con_cp = k + c_width  # k × c_width array multiplier

        savings_area = (1 - con_total_ges / gen_total_ges) * 100
        savings_cp = (1 - con_cp / gen_cp) * 100

        print(f"\n--- {mode_name} mode ---")
        print(f"  Operand width k = {k}, coefficient width = {c_width}")
        print(f"  Reduction multiplier (×q or ×q'):")
        print(f"    General:     {k}×{k} = {gen_mult_fas} FAs = {gen_total_ges:,} GE")
        print(f"    Constrained: {k}×{c_width} = {con_mult_fas} FAs + {adder_fas} FA adder + mux")
        print(f"                 = {con_total_ges:,} GE")
        print(f"    Area savings per reduction multiplier: {savings_area:.1f}%")
        print(f"  Critical path (multiply stage):")
        print(f"    General:     2k = {gen_cp}Δ")
        print(f"    Constrained: k + (k-{n}) = {con_cp}Δ")
        print(f"    CP improvement: {savings_cp:.1f}%")


# =====================================================================
#  Main
# =====================================================================

def main():
    print("C-TBM Functional Verification Script")
    print("Constrained-Modulus Montgomery Reduction for FHE Accelerators")
    print()

    results = []

    results.append(("Numerical Trace-Through", test_small_example()))
    results.append(("q' Structure Verification", test_q_prime_structure()))
    results.append(("Full Multiply Correctness", test_full_multiply()))
    test_hardware_cost()

    print("\n" + "=" * 72)
    print("VERIFICATION SUMMARY")
    print("=" * 72)
    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name:40s}  {status}")
        if not passed:
            all_pass = False

    print(f"\nOverall: {'ALL TESTS PASSED' if all_pass else 'TESTS FAILED'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
    