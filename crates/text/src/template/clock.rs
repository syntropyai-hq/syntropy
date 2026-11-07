// SPDX-License-Identifier: MIT OR Apache-2.0

//! `strftime_now(format)`, the date helper some templates call to write today's date. The
//! renderer never reads a clock: the caller passes the date in [`super::RenderRequest::date`],
//! so a rendering is a function of its inputs. The C-locale directives a date can fill are
//! supported: `%d %e %m %Y %y %b %B %h %a %A %j %%`. Any other directive is an error.

use std::fmt::Write as _;

use super::Date;

const MONTHS: [&str; 12] = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
];
const DAYS: [&str; 7] = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
];

fn leap(y: i32) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}

/// Days since 1970-01-01 (proleptic Gregorian).
fn days_from_civil(y: i32, m: u32, d: u32) -> i64 {
    let y = i64::from(y) - i64::from(m <= 2);
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let m = i64::from(m);
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + i64::from(d) - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

impl Date {
    /// Whether the date exists.
    pub(super) fn valid(self) -> bool {
        let days = [
            31,
            if leap(self.year) { 29 } else { 28 },
            31,
            30,
            31,
            30,
            31,
            31,
            30,
            31,
            30,
            31,
        ];
        (1..=12).contains(&self.month)
            && self.day >= 1
            && self.day <= days[usize::from(self.month - 1)]
    }

    fn weekday(self) -> usize {
        // 1970-01-01 was a Thursday (index 3 from Monday).
        let d = days_from_civil(self.year, u32::from(self.month), u32::from(self.day));
        (d + 3).rem_euclid(7) as usize
    }

    fn ordinal(self) -> i64 {
        days_from_civil(self.year, u32::from(self.month), u32::from(self.day))
            - days_from_civil(self.year, 1, 1)
            + 1
    }
}

/// Formats `date` with `format`.
///
/// # Errors
///
/// The directive that is not supported.
pub(super) fn strftime(date: Date, format: &str) -> Result<String, String> {
    let mut out = String::new();
    let mut chars = format.chars();
    let month = MONTHS[usize::from(date.month - 1)];
    let day = DAYS[date.weekday()];
    while let Some(c) = chars.next() {
        if c != '%' {
            out.push(c);
            continue;
        }
        // `write!` to a `String` cannot fail.
        let _ = match chars.next() {
            Some('d') => write!(out, "{:02}", date.day),
            Some('e') => write!(out, "{:>2}", date.day),
            Some('m') => write!(out, "{:02}", date.month),
            Some('Y') => write!(out, "{}", date.year),
            Some('y') => write!(out, "{:02}", date.year.rem_euclid(100)),
            Some('b' | 'h') => write!(out, "{}", &month[..3]),
            Some('B') => write!(out, "{month}"),
            Some('a') => write!(out, "{}", &day[..3]),
            Some('A') => write!(out, "{day}"),
            Some('j') => write!(out, "{:03}", date.ordinal()),
            Some('%') => write!(out, "%"),
            Some(other) => return Err(format!("strftime_now: `%{other}` is not supported")),
            None => return Err("strftime_now: a trailing `%`".to_owned()),
        };
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_like_the_c_locale() {
        let d = Date {
            year: 2026,
            month: 11,
            day: 7,
        };
        assert_eq!(strftime(d, "%d %b %Y").unwrap(), "07 Nov 2026");
        assert_eq!(
            strftime(d, "%A %B %e, %y %j %%").unwrap(),
            "Saturday November  7, 26 311 %"
        );
        assert!(strftime(d, "%H").is_err());
        assert!(strftime(d, "%").is_err());
        let leap_day = Date {
            year: 2024,
            month: 2,
            day: 29,
        };
        assert!(leap_day.valid());
        assert_eq!(strftime(leap_day, "%a %j").unwrap(), "Thu 060");
        assert!(
            !Date {
                year: 2023,
                month: 2,
                day: 29
            }
            .valid()
        );
    }
}
