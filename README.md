# C-TBM Functional Verification Script

Verification suite for the C-TBM (Constrained-Modulus Tunable-Bit Multiplier) architecture proposed in:

**"C-TBM: A Constrained-Modulus Montgomery Reduction Unit for Precision-Reconfigurable NTT Butterfly Architectures in FHE Hardware Accelerators"**
ECE 5560: Advanced Hardware Architecture Design Techniques
## Overview

This script verifies that the constrained-modulus Montgomery reduction decomposition for Proth primes (q = c · 2ⁿ + 1) produces bitwise-identical results to standard Montgomery reduction (REDC). It validates both the Step 3 (×q) and Step 2 (×q') decompositions.

## Tests

- **Test 1:** Step-by-step numerical trace using q = 97 = 3·2⁵ + 1
- **Test 2:** q' structure verification across 16 Proth primes (36-bit and 60-bit)
- **Test 3:** 168 full Montgomery multiplications comparing standard vs. constrained REDC
- **Test 4:** Hardware cost summary (area and critical path estimates)

## Requirements for Running

- Python 3.8+
- sympy (`pip install sympy`)
