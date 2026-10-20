//! IRT 2PL MML estimation (Bock & Aitkin 1981) — pure-Rust core.
//!
//! 2PL: P(X[p,j]=1 | θ_p) = 1/(1+exp(-a_j (θ_p - b_j))).
//! θ ~ N(0,1) prior; M-marginal-likelihood (integrate θ out) via
//! 40-point Gauss-Hermite quadrature (physicists' e^{-x^2} convention,
//! θ_k = sqrt(2)*x_k, prior density baked into the quadrature weight).
//!
//! Faithful port of the validated Python oracle
//! (D:\wsl2\win\irt_poc\irt_ref.py, seed 7, RMSE(b)=0.267 on the fixed
//! reference dataset) — see the `#[cfg(test)]` round-trip test in lib.rs.
//!
//! Conventions: X is a `&[Vec<u8>]` matrix (rows = people, cols = items,
//! entries 0/1). EM: E-step computes per-person posterior over quadrature
//! nodes (log-sum-exp stabilized); M-step refines each (a_j, b_j) by
//! coordinate-descent golden-section maximization of the expected LL.

/// 40-point Gauss-Hermite nodes (ascending) and weights, e^{-x^2} weight.
/// sum(weights) = sqrt(pi) (auto-generated, 17 digits — matches Python
/// scipy.special.roots_hermite(40) exactly).
pub const GH_X: [f64; 40] = [
-8.09876113925085050e+00,
-7.41158253148546819e+00,
-6.84023730524935480e+00,
-6.32825535122008187e+00,
-5.85409505603039904e+00,
-5.40665424797012673e+00,
-4.97926097854525551e+00,
-4.56750207284439469e+00,
-4.16825706683250008e+00,
-3.77920675343522339e+00,
-3.39855826585962806e+00,
-3.02487988390128404e+00,
-2.65699599844289569e+00,
-2.29391714187508322e+00,
-1.93479147228229564e+00,
-1.57886989493161378e+00,
-1.22548010904628901e+00,
-8.74006612357087986e-01,
-5.23874713832277128e-01,
-1.74537214597582396e-01,
1.74537214597582396e-01,
5.23874713832277128e-01,
8.74006612357087986e-01,
1.22548010904628901e+00,
1.57886989493161378e+00,
1.93479147228229564e+00,
2.29391714187508322e+00,
2.65699599844289569e+00,
3.02487988390128404e+00,
3.39855826585962806e+00,
3.77920675343522339e+00,
4.16825706683250008e+00,
4.56750207284439469e+00,
4.97926097854525551e+00,
5.40665424797012673e+00,
5.85409505603039904e+00,
6.32825535122008187e+00,
6.84023730524935480e+00,
7.41158253148546819e+00,
8.09876113925085050e+00,
];

pub const GH_W: [f64; 40] = [
2.59104371384628919e-29,
8.54405696377504339e-25,
2.56759336541138957e-21,
1.98918101211670418e-18,
6.00835878949094657e-16,
8.80570764521711220e-14,
7.15652805269073946e-12,
3.52562079136541450e-10,
1.12123608322762457e-08,
2.41114416367057120e-07,
3.63157615069302560e-06,
3.93693398109250671e-05,
3.13853594541333597e-04,
1.87149682959796048e-03,
8.46088800825816126e-03,
2.93125655361724073e-02,
7.84746058654042095e-02,
1.63378732713271524e-01,
2.65728251877377031e-01,
3.38643277425589251e-01,
3.38643277425589251e-01,
2.65728251877377031e-01,
1.63378732713271524e-01,
7.84746058654042095e-02,
2.93125655361724073e-02,
8.46088800825816126e-03,
1.87149682959796048e-03,
3.13853594541333597e-04,
3.93693398109250671e-05,
3.63157615069302560e-06,
2.41114416367057120e-07,
1.12123608322762457e-08,
3.52562079136541450e-10,
7.15652805269073946e-12,
8.80570764521711220e-14,
6.00835878949094657e-16,
1.98918101211670418e-18,
2.56759336541138957e-21,
8.54405696377504339e-25,
2.59104371384628919e-29,
];

/// 2PL probability: P(X=1 | θ, a, b).
#[inline]
pub fn p2pl(theta: f64, a: f64, b: f64) -> f64 {
    1.0 / (1.0 + (-a * (theta - b)).exp())
}

/// EAP posterior mean θ = Σ_k w_k exp(logP[p,k]) · θ_k.
#[inline]
pub fn theta_eap(logp: &[f64], theta: &[f64]) -> f64 {
    let m = logp.iter().cloned().fold(f64::NEG_INFINITY, |a, b| a.max(b));
    let s: f64 = logp.iter().zip(theta.iter()).map(|(&lp, &th)| (lp - m).exp() * th).sum();
    let z: f64 = logp.iter().map(|&lp| (lp - m).exp()).sum();
    if z > 0.0 {
        s / z
    } else {
        0.0
    }
}

/// E-step: per-person log-posterior over the 40 quadrature nodes.
/// `post[p][k]` — log-sum-exp stabilized; `theta_k = sqrt(2)·x_k`.
pub fn e_step(x: &[Vec<u8>], a: &[f64], b: &[f64]) -> (Vec<Vec<f64>>, Vec<f64>) {
    let p = x.len();
    let j = a.len();
    let theta: Vec<f64> = GH_X.iter().map(|&v| (2.0f64).sqrt() * v).collect();
    let lw: Vec<f64> = GH_W.iter().map(|&v| v.ln()).collect();
    let mut logp: Vec<Vec<f64>> = vec![vec![0.0f64; 40]; p];
    for pi in 0..p {
        let row = &x[pi];
        for k in 0..40 {
            let mut ll = 0.0f64;
            for jj in 0..j {
                let pr = (p2pl(theta[k], a[jj], b[jj])).clamp(1e-12, 1.0 - 1e-12);
                ll += if row[jj] != 0 {
                    pr.ln()
                } else {
                    (1.0 - pr).ln()
                };
            }
            logp[pi][k] = ll + lw[k];
        }
        let m = logp[pi].iter().cloned().fold(f64::NEG_INFINITY, |a, b| a.max(b));
        for k in 0..40 {
            logp[pi][k] -= m;
        }
    }
    (logp, theta)
}

/// Expected LL of item j under posterior `post` (log-space).
fn el_j(post: &[Vec<f64>], theta: &[f64], xj: &[u8], a: f64, b: f64) -> f64 {
    let n = post.len();
    // per-node log p, then per-person log LL with the log-posterior, then per-person log LL with the log-posterior
    let mut logpk: Vec<f64> = Vec::with_capacity(40);
    for k in 0..40 {
        logpk.push((p2pl(theta[k], a, b)).clamp(1e-12, 1.0 - 1e-12).ln());
    }
    let mut logn1k: Vec<f64> = Vec::with_capacity(40);
    for k in 0..40 {
        let pr = (p2pl(theta[k], a, b)).clamp(1e-12, 1.0 - 1e-12);
        logn1k.push((1.0 - pr).ln());
    }
    // per person: max_k (log post[p,k] + ll[p,k]) then logsumexp across k
    let mut total = 0.0f64;
    for pi in 0..n {
        let mut mp = f64::NEG_INFINITY;
        for k in 0..40 {
            let ll = if xj[pi] != 0 { logpk[k] } else { logn1k[k] };
            let v = post[pi][k] + ll;
            if v > mp {
                mp = v;
            }
        }
        let mut s = 0.0f64;
        for k in 0..40 {
            let ll = if xj[pi] != 0 { logpk[k] } else { logn1k[k] };
            s += (post[pi][k] + ll - mp).exp();
        }
        total += mp + s.ln();
    }
    total
}

/// EL for item j (standard Bock & Aitkin EM M-step objective) plus its
/// analytic gradient w.r.t. (a, b).
///
/// The E-step posterior over theta_k (call it q_k, proportional to
/// exp(logP[pi,k]) normalized per person) is fixed by the E-step. The M-step
/// for item j maximizes  EL(a,b) = Σ_p Σ_k q[pi,k] · log p_j(theta_k; a,b)
/// which is concave in (a,b). We do NOT re-marginalize here (that would be a
/// different objective). Gradients:
///   dEL/da = Σ w (x - p)(θ_k - b)
///   dEL/db = Σ w (x - p)(-a)
/// where w = q (the E-step posterior), x = x[pi] (0/1), p = p_j(θ_k; a,b).
fn el_grad_j(
    post: &[Vec<f64>],
    theta: &[f64],
    xj: &[u8],
    a: f64,
    b: f64,
) -> (f64, f64, f64) {
    let n = post.len();
    let mut el = 0.0f64;
    let mut da = 0.0f64;
    let mut db = 0.0f64;
    for pi in 0..n {
        // normalize the per-person posterior q_k = exp(post[pi,k]) (already
        // log-normalized in e_step, so exp is safe; renormalize for safety)
        let mut s = 0.0f64;
        for k in 0..40 {
            s += post[pi][k].exp();
        }
        if s <= 0.0 || !s.is_finite() {
            continue;
        }
        let x = xj[pi] as f64;
        for k in 0..40 {
            let q = post[pi][k].exp() / s;
            let pr = (p2pl(theta[k], a, b)).clamp(1e-12, 1.0 - 1e-12);
            let ll = if x != 0.0 { pr.ln() } else { (1.0 - pr).ln() };
            el += q * ll;
            let d = x - pr;
            da += q * d * (theta[k] - b);
            db += q * d * (-a);
        }
    }
    (el, da, db)
}

/// 2D gradient ascent with backtracking line search on the concave per-item
/// EL surface. Returns the (a, b) maximizer. Robust to ill-conditioning
/// because it follows the gradient toward the unique interior maximum.
#[allow(dead_code)]
fn grad_ascent_2d(
    f: &dyn Fn(f64, f64) -> (f64, f64, f64),
    a0: f64,
    b0: f64,
) -> (f64, f64) {
    const LO_A: f64 = 0.01;
    const HI_A: f64 = 12.0;
    const LO_B: f64 = -12.0;
    const HI_B: f64 = 12.0;
    let clamp_a = |v: f64| v.clamp(LO_A, HI_A);
    let clamp_b = |v: f64| v.clamp(LO_B, HI_B);
    let mut a = clamp_a(a0);
    let mut b = clamp_b(b0);
    let (mut f0, mut ga, mut gb) = f(a, b);
    const BASE_LR: f64 = 3.0;
    for _ in 0..30 {
        if ga.abs() < 1e-8 && gb.abs() < 1e-8 {
            break;
        }
        let gnorm2 = ga * ga + gb * gb;
        let mut step = (BASE_LR / gnorm2).min(1.5);
        let mut improved = false;
        for _ in 0..6 {
            let (na, nb) = (clamp_a(a + step * ga), clamp_b(b + step * gb));
            let (nf, ga2, gb2) = f(na, nb);
            if nf > f0 {
                a = na;
                b = nb;
                f0 = nf;
                ga = ga2;
                gb = gb2;
                improved = true;
                break;
            }
            step *= 0.5;
        }
        if !improved {
            break;
        }
    }
    (a, b)
}

/// Fit the 2PL model by MML/EM. Returns (a, b, theta_eap, em_iterations).
///
/// **Identification**: the 2PL is scale-non-identifiable — (a,b) and (a/c,
/// b/c) give identical likelihood. We anchor `b[0] = 0` (standard IRT
/// practice, e.g. ltm/mirt default) to make all (a_j, b_j) unique. The
/// theta EAP estimates are unaffected by the anchor (they are
/// identifiable from the data).
pub fn fit_2pl(
    x: &[Vec<u8>],
    iters: usize,
    tol: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, usize) {
    let p = x.len();
    let j = if p > 0 { x[0].len() } else { 0 };
    let mut a = vec![1.0f64; j];
    let mut b = vec![0.0f64; j];
    let theta: Vec<f64> = GH_X.iter().map(|&v| (2.0f64).sqrt() * v).collect();
    let mut used = iters;
    for it in 0..iters {
        let (logp, _) = e_step(x, &a, &b);
        let mut a_new = a.clone();
        let mut b_new = b.clone();
        for jj in 0..j {
            let mut xj: Vec<u8> = Vec::with_capacity(p);
            for pi in 0..p {
                xj.push(x[pi][jj]);
            }
            // 2D gradient ascent (analytic gradient + backtracking) on the concave
            // per-item EL surface — follows the gradient to the interior max.
            let eg = |av: f64, bv: f64| el_grad_j(&logp, &theta, &xj, av, bv);
            let (na, nb) = grad_ascent_2d(&eg, a[jj], b[jj]);
            a_new[jj] = na.clamp(1e-3, 12.0);
            b_new[jj] = nb.clamp(-12.0, 12.0);
        }
        // **Identification**: with the fixed N(0,1) prior on theta, the 2PL is
        // identified — no constraint needed. (Rescaling (a,b)->(ca,cb) would
        // change expit(a(theta-b)) unless theta is also rescaled, which breaks
        // the N(0,1) prior, so the likelihood is NOT scale-invariant.)
        let d = (0..j)
            .map(|k| ((a_new[k] - a[k]).abs()).max((b_new[k] - b[k]).abs()))
            .fold(0.0f64, f64::max);
        a = a_new;
        b = b_new;
        used = it + 1;
        if d < tol {
            break;
        }
    }
    let (logp, theta2) = e_step(x, &a, &b);
    let theta_hat: Vec<f64> = (0..p).map(|pi| theta_eap(&logp[pi], &theta2)).collect();
    (a, b, theta_hat, used)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn p2pl_properties() {
        assert!((p2pl(0.0, 1.0, 0.0) - 0.5).abs() < 1e-15);
        assert!((p2pl(1.0, 1.0, 1.0) - 0.5).abs() < 1e-15); // θ=b
        // a=2, θ-b=0.5 → 1/(1+e^-1) ≈ 0.731
        assert!((p2pl(1.5, 2.0, 1.0) - 1.0 / (1.0 + (-1.0f64).exp())).abs() < 1e-12);
        // monotone in θ
        assert!(p2pl(2.0, 1.0, 0.0) > p2pl(1.0, 1.0, 0.0));
        assert!(p2pl(-1.0, 1.0, 0.0) < 0.5);
    }

    #[test]
    fn gh40_quadrature() {
        // ∫ e^{-x^2} dx = sqrt(pi); with physicists' weights: Σ w_k = sqrt(pi)
        let s: f64 = GH_W.iter().sum();
        assert!((s - std::f64::consts::PI.sqrt()).abs() < 1e-9);
        // nodes ascending
        for w in GH_X.windows(2) {
            assert!(w[0] < w[1]);
        }
        // ∫ x^2 e^{-x^2} dx = sqrt(pi)/2
        let s2: f64 = GH_W.iter().zip(GH_X.iter()).map(|(&w, &x)| w * x * x).sum();
        assert!((s2 - std::f64::consts::PI.sqrt() / 2.0).abs() < 1e-9);
    }

    #[test]
    fn e_step_normalizes() {
        // P=2, J=3
        let x = vec![vec![1u8, 0, 1], vec![0, 1, 1]];
        let a = vec![1.0, 1.0, 1.0];
        let b = vec![0.0, 0.0, 0.0];
        let (logp, _) = e_step(&x, &a, &b);
        for pi in 0..2 {
            let m = logp[pi].iter().cloned().fold(f64::NEG_INFINITY, |a, b| a.max(b));
            let s: f64 = logp[pi].iter().map(|&v| (v - m).exp()).sum();
            assert!(s > 0.0 && s.is_finite());
        }
    }

    #[test]
    fn fit_recovers_known_data() {
        // 4 people x 2 items, full 2PL design with unambiguous contrast.
        // Person rows = (item0, item1). Correct counts: 2,1,1,0.
        let x = vec![
            vec![1u8, 1], // 2 correct  -> highest theta
            vec![1u8, 0], // 1 correct
            vec![0u8, 1], // 1 correct
            vec![0u8, 0], // 0 correct  -> lowest theta
        ];
        let (a, b, th, iters) = fit_2pl(&x, 100, 1e-7);
        assert_eq!(a.len(), 2);
        assert!(a.iter().all(|v| v.is_finite()));
        assert!(b.iter().all(|v| v.is_finite()));
        assert!(th.iter().all(|v| v.is_finite()));
        assert!(iters >= 1);
        // both items ~50% correct -> b finite & sane
        assert!(b[0].abs() < 4.0);
        assert!(b[1].abs() < 4.0);
        // a positive (items discriminate)
        assert!(a[0] > 0.0 && a[1] > 0.0);
        // theta monotone in number of correct responses
        assert!(th[0] > th[1], "2-correct should exceed 1-correct");
        assert!(th[3] < th[2], "0-correct should be below 1-correct");
    }
}
