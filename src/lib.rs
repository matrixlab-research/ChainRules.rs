//! Backend-agnostic contracts for custom automatic-differentiation rules.
//!
//! This crate is intentionally not an AD engine. It defines a small common
//! protocol that numerical packages and AD backends can implement without
//! depending on one another.

#![forbid(unsafe_code)]

/// Associates a primal value with the type used for its tangent or cotangent.
pub trait Differentiable {
    /// The tangent-space representation for this primal type.
    type Tangent;
}

/// Marks a value that has no tangent space, such as a discrete configuration.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct NoTangent;

/// Represents a structurally known zero without allocating a full tangent.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ZeroTangent;

impl Differentiable for f32 {
    type Tangent = f32;
}

impl Differentiable for f64 {
    type Tangent = f64;
}

impl Differentiable for () {
    type Tangent = NoTangent;
}

impl<A, B> Differentiable for (A, B)
where
    A: Differentiable,
    B: Differentiable,
{
    type Tangent = (A::Tangent, B::Tangent);
}

impl<T, const N: usize> Differentiable for [T; N]
where
    T: Differentiable,
{
    type Tangent = [T::Tangent; N];
}

/// A forward-mode rule computing a primal output and a Jacobian-vector product.
pub trait JvpRule<Args>
where
    Args: Differentiable,
{
    /// The primal output type.
    type Output: Differentiable;

    /// Evaluates the primal operation and propagates an input tangent forward.
    fn jvp(
        &self,
        args: &Args,
        tangent: &Args::Tangent,
    ) -> (
        Self::Output,
        <Self::Output as Differentiable>::Tangent,
    );
}

/// A reverse-mode pullback from an output cotangent to input cotangents.
pub trait Pullback<OutputTangent, InputTangent> {
    /// Applies the pullback once.
    fn apply(self, cotangent: OutputTangent) -> InputTangent;
}

impl<F, OutputTangent, InputTangent> Pullback<OutputTangent, InputTangent> for F
where
    F: FnOnce(OutputTangent) -> InputTangent,
{
    fn apply(self, cotangent: OutputTangent) -> InputTangent {
        self(cotangent)
    }
}

/// A reverse-mode rule computing a primal output and a reusable pullback object.
///
/// The generic associated type lets a pullback borrow inputs or caches from the
/// forward evaluation instead of forcing large scientific arrays to be cloned.
pub trait VjpRule<Args>
where
    Args: Differentiable,
{
    /// The primal output type.
    type Output: Differentiable;

    /// Pullback produced for one primal evaluation.
    type Pullback<'a>: Pullback<
        <Self::Output as Differentiable>::Tangent,
        Args::Tangent,
    >
    where
        Self: 'a,
        Args: 'a;

    /// Evaluates the primal operation and constructs its pullback.
    fn vjp<'a>(&'a self, args: &'a Args) -> (Self::Output, Self::Pullback<'a>);
}

/// Small validation helpers for custom rules.
pub mod testing {
    /// Estimates a scalar directional derivative with a central difference.
    #[must_use]
    pub fn central_difference<F>(f: F, x: f64, direction: f64, step: f64) -> f64
    where
        F: Fn(f64) -> f64,
    {
        assert!(step > 0.0, "finite-difference step must be positive");
        (f(x + step * direction) - f(x - step * direction)) / (2.0 * step)
    }

    /// Asserts approximate equality using combined absolute and relative tolerances.
    pub fn assert_close(actual: f64, expected: f64, relative: f64, absolute: f64) {
        let tolerance = absolute + relative * actual.abs().max(expected.abs());
        assert!(
            (actual - expected).abs() <= tolerance,
            "values differ: actual={actual}, expected={expected}, tolerance={tolerance}"
        );
    }

    /// Checks the defining adjoint identity `<Jv, w> = <v, J^T w>` for scalars.
    pub fn assert_scalar_adjoint_identity(
        jvp: f64,
        output_cotangent: f64,
        input_tangent: f64,
        input_cotangent: f64,
        tolerance: f64,
    ) {
        assert_close(
            jvp * output_cotangent,
            input_tangent * input_cotangent,
            tolerance,
            tolerance,
        );
    }
}
#[cfg(test)]
mod tests {
    use super::{JvpRule, Pullback, VjpRule};
    use crate::testing::{
        assert_close, assert_scalar_adjoint_identity, central_difference,
    };

    struct Square;

    struct SquarePullback {
        x: f64,
    }

    impl Pullback<f64, f64> for SquarePullback {
        fn apply(self, cotangent: f64) -> f64 {
            2.0 * self.x * cotangent
        }
    }

    impl JvpRule<f64> for Square {
        type Output = f64;

        fn jvp(&self, x: &f64, tangent: &f64) -> (f64, f64) {
            (x * x, 2.0 * x * tangent)
        }
    }

    impl VjpRule<f64> for Square {
        type Output = f64;
        type Pullback<'a> = SquarePullback;

        fn vjp<'a>(&'a self, x: &'a f64) -> (f64, Self::Pullback<'a>) {
            (x * x, SquarePullback { x: *x })
        }
    }

    #[test]
    fn jvp_matches_finite_difference() {
        let x = 3.0;
        let direction = -0.25;
        let (_, jvp) = Square.jvp(&x, &direction);
        let numerical = central_difference(|value| value * value, x, direction, 1.0e-5);

        assert_close(jvp, numerical, 1.0e-9, 1.0e-9);
    }

    #[test]
    fn vjp_satisfies_adjoint_identity() {
        let x = 3.0;
        let input_tangent = -0.25;
        let output_cotangent = 1.75;
        let (_, jvp) = Square.jvp(&x, &input_tangent);
        let (_, pullback) = Square.vjp(&x);
        let input_cotangent = pullback.apply(output_cotangent);

        assert_scalar_adjoint_identity(
            jvp,
            output_cotangent,
            input_tangent,
            input_cotangent,
            1.0e-12,
        );
    }
}
