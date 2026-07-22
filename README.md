# ChainRules.rs

`ChainRules.rs` is a minimal, backend-agnostic protocol for custom automatic-
differentiation rules in scientific Rust.

The project separates three concerns:

1. scientific packages own primal operations such as matrix multiplication,
   linear solves, FFTs, and differential-equation solves;
2. rule packages expose efficient Jacobian-vector products (JVPs) and
   vector-Jacobian products (VJPs);
3. AD backends consume those rules or fall back to generated differentiation.

## Status

This repository is an experimental protocol, not a production AD framework and
not a feature-complete port of Julia's ChainRules.jl.

The current crate deliberately does **not**:

- build or execute computation graphs;
- implement an automatic-differentiation engine;
- require nightly Rust or `std::autodiff`;
- define one universal array or tensor type;
- use a runtime-global rule registry.

## Core protocol

The narrow waist consists of four concepts:

- `Differentiable`: maps a primal type to its tangent representation;
- `JvpRule`: evaluates a primal operation and propagates an input tangent;
- `VjpRule`: evaluates a primal operation and returns a pullback;
- `testing`: validates rules with finite differences and adjoint identities.

```rust
use chainrules_core::{Differentiable, JvpRule};

struct Square;

impl JvpRule<f64> for Square {
    type Output = f64;

    fn jvp(&self, x: &f64, dx: &f64) -> (f64, f64) {
        (x * x, 2.0 * x * dx)
    }
}

let (value, directional_derivative) = Square.jvp(&3.0, &0.5);
assert_eq!(value, 9.0);
assert_eq!(directional_derivative, 3.0);
```

## Design principles

- Prefer explicit, statically dispatched rules over a global runtime registry.
- Keep the core independent from `ndarray`, `faer`, `nalgebra`, and tensor
  frameworks; integrations belong in package-owned implementations or adapters.
- Treat mutating operations as a separate semantic problem rather than hiding
  aliasing behind a pure-function interface.
- Validate behavior, not only API shape: every rule should pass Taylor or finite-
  difference checks and the identity `<Jv, w> = <v, J^T w>`.
- Keep experimental compiler integration behind optional adapter crates.

## Reference roadmap

The first useful milestone is two end-to-end scientific rules:

1. dense matrix multiplication, covering structured tangents and transpose or
   conjugate-transpose pullbacks;
2. linear solve `Ax = b`, covering implicit differentiation and factorization
   reuse.

An ODE-solve rule is the next workflow-level reference. Adapters for dual-number
forward mode, tensor AD, and nightly `std::autodiff` should be added only after
the core rule semantics and conformance tests are demonstrated by these
workflows.

## License

MIT
