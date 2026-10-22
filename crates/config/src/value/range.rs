// SPDX-License-Identifier: MIT OR Apache-2.0

//! `InRange`: how a key's `min` and `max` apply to its type.

use std::cmp::Ordering;
use std::fmt::Display;

use serde::de::DeserializeOwned;
use syntropy_core::policy::{Fraction, Reserve};

use super::{ByteSize, Duration, Rate, ValueError};

/// A type a key can declare `min` or `max` for. The bounds are written in the attribute as
/// values of [`InRange::Bound`].
pub trait InRange {
    /// The type the bounds are written in.
    type Bound: DeserializeOwned + Display;

    /// Checks `self` against the bounds.
    fn check_range(
        &self,
        min: Option<&Self::Bound>,
        max: Option<&Self::Bound>,
    ) -> Result<(), ValueError>;
}

/// Checks `value` against bounds of the same, ordered type.
fn ordered<T: PartialOrd + Display>(
    value: &T,
    min: Option<&T>,
    max: Option<&T>,
) -> Result<(), ValueError> {
    if let Some(min) = min
        && !matches!(
            value.partial_cmp(min),
            Some(Ordering::Greater | Ordering::Equal)
        )
    {
        return Err(ValueError::out_of_range(format!(
            "{value} is below the minimum, {min}"
        )));
    }
    if let Some(max) = max
        && !matches!(
            value.partial_cmp(max),
            Some(Ordering::Less | Ordering::Equal)
        )
    {
        return Err(ValueError::out_of_range(format!(
            "{value} is above the maximum, {max}"
        )));
    }
    Ok(())
}

macro_rules! ordered_in_range {
    ($($ty:ty),*) => {$(
        impl InRange for $ty {
            type Bound = $ty;

            fn check_range(&self, min: Option<&$ty>, max: Option<&$ty>) -> Result<(), ValueError> {
                ordered(self, min, max)
            }
        }
    )*};
}

ordered_in_range!(u8, u16, u32, u64, usize, i8, i16, i32, i64, f32, f64);
ordered_in_range!(ByteSize, Duration);

/// A range on a fraction is written as numbers in [0, 1], such as `max = 0.5`.
impl InRange for Fraction {
    type Bound = f32;

    fn check_range(&self, min: Option<&f32>, max: Option<&f32>) -> Result<(), ValueError> {
        ordered(&self.get(), min, max)
    }
}

/// A range on a rate needs bounds of the same kind: tokens for a token rate, bytes for a byte
/// rate.
impl InRange for Rate {
    type Bound = Rate;

    fn check_range(&self, min: Option<&Rate>, max: Option<&Rate>) -> Result<(), ValueError> {
        let same_kind = |bound: &Rate| {
            if std::mem::discriminant(bound) == std::mem::discriminant(self) {
                Ok(())
            } else {
                Err(ValueError::out_of_range(format!(
                    "{self} cannot be compared with the bound {bound}: one counts tokens, the \
                     other bytes"
                )))
            }
        };
        min.map_or(Ok(()), same_kind)?;
        max.map_or(Ok(()), same_kind)?;
        ordered(self, min, max)
    }
}

/// A range on a reserve bounds its floor, `min`, so `min = "1GiB"` guarantees the reserve
/// resolves to at least 1 GiB on any tier.
impl InRange for Reserve {
    type Bound = ByteSize;

    fn check_range(
        &self,
        min: Option<&ByteSize>,
        max: Option<&ByteSize>,
    ) -> Result<(), ValueError> {
        ordered(&ByteSize(self.min), min, max)
            .map_err(|e| ValueError::out_of_range(format!("the reserve's floor: {e}")))
    }
}

/// An unset value is in range.
impl<T: InRange> InRange for Option<T> {
    type Bound = T::Bound;

    fn check_range(
        &self,
        min: Option<&T::Bound>,
        max: Option<&T::Bound>,
    ) -> Result<(), ValueError> {
        self.as_ref().map_or(Ok(()), |v| v.check_range(min, max))
    }
}

/// Every element must be in range.
impl<T: InRange> InRange for Vec<T> {
    type Bound = T::Bound;

    fn check_range(
        &self,
        min: Option<&T::Bound>,
        max: Option<&T::Bound>,
    ) -> Result<(), ValueError> {
        self.iter().try_for_each(|v| v.check_range(min, max))
    }
}

#[cfg(test)]
mod tests {
    use syntropy_core::Bytes;

    use super::*;

    #[test]
    fn checks_ordered_types() {
        assert!(5u32.check_range(Some(&1), Some(&8)).is_ok());
        assert!(0u32.check_range(Some(&1), None).is_err());
        assert!(9u32.check_range(None, Some(&8)).is_err());
        assert!(f64::NAN.check_range(Some(&0.0), None).is_err());
        let gib = ByteSize::gib(1);
        assert!(ByteSize::mib(512).check_range(Some(&gib), None).is_err());
        assert!(
            Some(ByteSize::mib(512))
                .check_range(None, Some(&gib))
                .is_ok()
        );
        assert!(None::<ByteSize>.check_range(Some(&gib), None).is_ok());
        assert!(vec![1u8, 9].check_range(None, Some(&8)).is_err());
    }

    #[test]
    fn a_reserve_range_bounds_its_floor() {
        let gib = ByteSize::gib(1);
        let reserve = Reserve {
            min: Bytes(6 << 30),
            fraction: 0.1,
            max: None,
        };
        assert!(reserve.check_range(Some(&gib), None).is_ok());
        let share_only = Reserve {
            min: Bytes(0),
            fraction: 0.1,
            max: None,
        };
        assert!(share_only.check_range(Some(&gib), None).is_err());
    }

    #[test]
    fn a_rate_range_needs_one_kind() {
        let tokens = Rate::Tokens(5.0);
        assert!(tokens.check_range(Some(&Rate::Tokens(1.0)), None).is_ok());
        assert!(tokens.check_range(Some(&Rate::Tokens(6.0)), None).is_err());
        let bytes = Rate::Bytes(syntropy_core::BytesPerSec(1.0));
        assert!(tokens.check_range(Some(&bytes), None).is_err());
    }
}
