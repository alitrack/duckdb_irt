//! duckdb_irt — 2PL Item Response Theory via MML/EM (Bock & Aitkin 1981).
//!
//! Function:
//!   irt_2pl_param(x VARCHAR, j BIGINT, which VARCHAR, idx BIGINT,
//!                 iters BIGINT, tol DOUBLE) → DOUBLE
//!
//!   x     : flat response matrix as a string, people-major:
//!           "1,0,1, 0,1,1, 1,1,0" (commas, spaces, or newlines between values)
//!           P = len/ j, so the string holds exactly P*j values.
//!   j     : items per person (BIGINT)
//!   which : 'a' → item discrimination a_j, 'b' → item difficulty b_j,
//!           'theta' → person ability θ_i (EAP under N(0,1) prior)
//!   idx   : 0-based index into the relevant set (j items / P people)
//!   iters : max EM iterations (BIGINT)
//!   tol   : convergence tolerance (DOUBLE)
//!
//! Example (300 people × 12 items, matrix stored in table `resp(person_id,
//! item_id, r)`):
//!   CREATE TABLE x AS
//!     SELECT string_agg(r::VARCHAR, ',') FROM resp
//!            ORDER BY person_id, item_id;
//!   SELECT irt_2pl_param(x.x, 12, 'a', 0, 200, 1e-7) AS item0_a FROM x;
//!
//! Math in `src/math.rs` — faithful Rust port of the validated Python oracle
//! (D:\wsl2\win\irt_poc\irt_ref.py; 300x12 reference dataset, seed 7,
//! RMSE(b)=0.267).

pub mod math;

use quack_rs::{
    entry_point::init_extension_v2,
    prelude::*,
    vector::{VectorReader, VectorWriter},
};
use libduckdb_sys::{
    duckdb_data_chunk, duckdb_data_chunk_get_size, duckdb_extension_access,
    duckdb_extension_info, duckdb_vector,
};

use math::fit_2pl;

/// Parse a flat response string ("1,0,1 0,1,1\n...") into a Vec<u8> of 0/1.
/// Delimiters: whitespace, commas. Any other character (including a digit
/// other than 0/1) is an error, so typos are caught rather than silently
/// skipped.
fn parse_responses(s: &str) -> Result<Vec<u8>, ()> {
    let mut out: Vec<u8> = Vec::with_capacity(s.len() / 2);
    let mut tok = String::new();
    let flush = |out: &mut Vec<u8>, tok: &mut String| -> Result<(), ()> {
        if !tok.is_empty() {
            match tok.as_str() {
                "0" => out.push(0),
                "1" => out.push(1),
                _ => return Err(()),
            }
            tok.clear();
        }
        Ok(())
    };
    for c in s.chars() {
        if c.is_ascii_digit() {
            tok.push(c);
        } else if c.is_whitespace() || c == ',' {
            flush(&mut out, &mut tok)?;
        } else {
            return Err(());
        }
    }
    flush(&mut out, &mut tok)?;
    Ok(out)
}

unsafe extern "C" fn irt_param_fn(
    _info: libduckdb_sys::duckdb_function_info,
    input: duckdb_data_chunk,
    output: duckdb_vector,
) {
    let n = usize::try_from(unsafe { duckdb_data_chunk_get_size(input) }).unwrap_or(0);
    let rx = unsafe { VectorReader::new(input, 0) }; // x   VARCHAR
    let rj = unsafe { VectorReader::new(input, 1) }; // j   BIGINT
    let rw = unsafe { VectorReader::new(input, 2) }; // which VARCHAR
    let ri = unsafe { VectorReader::new(input, 3) }; // idx BIGINT
    let rit = unsafe { VectorReader::new(input, 4) }; // iters BIGINT
    let rt = unsafe { VectorReader::new(input, 5) }; // tol   DOUBLE
    let mut out = unsafe { VectorWriter::new(output) };
    for i in 0..n {
        if !unsafe { rx.is_valid(i) }
            || !unsafe { rj.is_valid(i) }
            || !unsafe { rw.is_valid(i) }
            || !unsafe { ri.is_valid(i) }
        {
            unsafe { out.set_null(i) };
            continue;
        }
        let s = unsafe { rx.read_str(i) };
        let j = unsafe { rj.read_i64(i) } as usize;
        let which = unsafe { rw.read_str(i) };
        let idx = unsafe { ri.read_i64(i) } as usize;
        let iters = if unsafe { rit.is_valid(i) } {
            (unsafe { rit.read_i64(i) }) as usize
        } else {
            200
        };
        let tol = if unsafe { rt.is_valid(i) } {
            unsafe { rt.read_f64(i) }
        } else {
            1e-7
        };
        if j < 2 || iters < 1 || !(tol > 0.0) {
            unsafe { out.set_null(i) };
            continue;
        }
        let flat = match parse_responses(s) {
            Ok(f) => f,
            Err(_) => {
                unsafe { out.set_null(i) };
                continue;
            }
        };
        if flat.len() % j != 0 {
            unsafe { out.set_null(i) };
            continue;
        }
        let p = flat.len() / j;
        if p < 2 {
            unsafe { out.set_null(i) };
            continue;
        }
        // build matrix
        let mut mat: Vec<Vec<u8>> = Vec::with_capacity(p);
        for pi in 0..p {
            mat.push(flat[pi * j..pi * j + j].to_vec());
        }
        let (a, b, theta, _it) = math::fit_2pl(&mat, iters, tol);
        let result: f64 = match which {
            "a" => a.get(idx).copied().unwrap_or(f64::NAN),
            "b" => b.get(idx).copied().unwrap_or(f64::NAN),
            "theta" => theta.get(idx).copied().unwrap_or(f64::NAN),
            _ => f64::NAN,
        };
        if result.is_finite() {
            unsafe { out.write_f64(i, result) };
        } else {
            unsafe { out.set_null(i) };
        }
    }
}

fn register_all(reg: &impl Registrar) -> Result<(), ExtensionError> {
    unsafe {
        reg.register_scalar(
            ScalarFunctionBuilder::new("irt_2pl_param")
                .param(TypeId::Varchar)
                .param(TypeId::BigInt)
                .param(TypeId::Varchar)
                .param(TypeId::BigInt)
                .param(TypeId::BigInt)
                .param(TypeId::Double)
                .returns(TypeId::Double)
                .function(irt_param_fn),
        )?;
    }
    Ok(())
}

#[no_mangle]
pub unsafe extern "C" fn irt_ext_init_c_api(
    info: duckdb_extension_info,
    access: *const duckdb_extension_access,
) -> bool {
    unsafe {
        init_extension_v2(info, access, quack_rs::DUCKDB_API_VERSION, |con| {
            register_all(con)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_basic() {
        assert_eq!(parse_responses("1,0,1").unwrap(), vec![1, 0, 1]);
        assert_eq!(parse_responses("1 0 1").unwrap(), vec![1, 0, 1]);
        assert_eq!(parse_responses("1\n0\n1").unwrap(), vec![1, 0, 1]);
        assert_eq!(parse_responses("1, 0, 1").unwrap(), vec![1, 0, 1]);
        assert!(parse_responses("1,2,1").is_err());
        assert!(parse_responses("1,a,1").is_err());
        assert!(parse_responses("").unwrap().is_empty());
    }

    #[test]
    fn p2pl_symmetry() {
        assert!((math::p2pl(0.0, 1.0, 0.0) - 0.5).abs() < 1e-15);
    }

    #[test]
    fn fit_smoke() {
        // 10 people × 3 items, deterministic
        let flat = vec![
            1u8, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1,
            0, 0, 1, 0, 1, 1,
        ];
        let j = 3usize;
        let p = flat.len() / j;
        let mut mat: Vec<Vec<u8>> = Vec::new();
        for pi in 0..p {
            mat.push(flat[pi * j..pi * j + j].to_vec());
        }
        let (a, b, theta, _it) = math::fit_2pl(&mat, 100, 1e-7);
        assert_eq!(a.len(), j);
        assert_eq!(b.len(), j);
        assert_eq!(theta.len(), p);
        assert!(a.iter().all(|v| v.is_finite()));
        assert!(b.iter().all(|v| v.is_finite()));
    }
}
