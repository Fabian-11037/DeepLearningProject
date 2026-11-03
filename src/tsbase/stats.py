"""Inferencia estadistica del Entregable 1."""
import itertools
import numpy as np
from scipy import stats


def dm_test(e1, e2, h=1, power=2):
    """Diebold-Mariano con correccion HLN. e1,e2: errores de pronostico (N,). H0: igual precision.
    Estadistico > 0 => el modelo 2 es mas preciso (perdida de 1 mayor)."""
    e1, e2 = np.asarray(e1, float), np.asarray(e2, float)
    d = np.abs(e1) ** power - np.abs(e2) ** power
    n = len(d); dbar = d.mean()
    dc = d - dbar
    gam = [np.dot(dc[k:], dc[:n - k]) / n for k in range(h)]
    var = (gam[0] + 2 * sum(gam[1:])) / n
    if var <= 0:
        return np.nan, np.nan
    dm = dbar / np.sqrt(var)
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = dm * hln
    p = 2 * (1 - stats.t.cdf(abs(stat), df=n - 1))
    return float(stat), float(p)


def holm(pvals):
    p = np.asarray(pvals, float); m = len(p)
    order = np.argsort(p); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i]); adj[i] = min(1.0, run)
    return adj


def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    gt = sum((x > y) for x in a for y in b); lt = sum((x < y) for x in a for y in b)
    return (gt - lt) / (len(a) * len(b))


def cohen_dz(diff):
    diff = np.asarray(diff, float)
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 0 else np.nan


def boot_ci(x, stat=np.mean, B=10000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed); x = np.asarray(x, float)
    bs = [stat(rng.choice(x, len(x), replace=True)) for _ in range(B)]
    return float(np.quantile(bs, alpha / 2)), float(np.quantile(bs, 1 - alpha / 2))


def t_ci(x, alpha=0.05):
    x = np.asarray(x, float); n = len(x)
    m, se = x.mean(), x.std(ddof=1) / np.sqrt(n)
    q = stats.t.ppf(1 - alpha / 2, n - 1)
    return float(m - q * se), float(m + q * se)


def required_runs(sigma_d, delta, alpha=0.05, power=0.80, m=1):
    """R ~ ((z_{1-a'/2}+z_{1-b}) * sigma_d / delta)^2 con a' = alpha/m (Bonferroni)."""
    z = stats.norm.ppf(1 - alpha / m / 2) + stats.norm.ppf(power)
    return float((z * sigma_d / delta) ** 2)


def min_detectable(sigma_d, R, alpha=0.05, power=0.80, m=1):
    z = stats.norm.ppf(1 - alpha / m / 2) + stats.norm.ppf(power)
    return float(z * sigma_d / np.sqrt(R))


def paired_report(a, b, alpha=0.05):
    """a, b: metricas por semilla (mismo indice = misma semilla). Devuelve dict (a - b)."""
    a, b = np.asarray(a, float), np.asarray(b, float); d = a - b
    out = dict(mean_diff=float(d.mean()), sd_diff=float(d.std(ddof=1)), n=len(d))
    out["t_ci95"] = t_ci(d); out["boot_ci95"] = boot_ci(d)
    out["shapiro_p"] = float(stats.shapiro(d)[1]) if len(d) >= 3 else np.nan
    out["t_stat"], out["t_p"] = [float(v) for v in stats.ttest_rel(a, b)]
    try:
        out["wilcoxon_p"] = float(stats.wilcoxon(a, b, method="auto")[1]) if len(d) >= 6 else np.nan
    except ValueError:
        out["wilcoxon_p"] = np.nan
    out["cohen_dz"] = cohen_dz(d); out["cliff_delta"] = float(cliffs_delta(a, b))
    out["prob_superiority"] = float(np.mean(d < 0) + 0.5 * np.mean(d == 0))  # P(a mejor que b)
    return out


def tost_paired(d, delta, alpha=0.05):
    """TOST pareado: H0 |mu_d| >= delta. Devuelve p (max de las dos unilaterales)."""
    d = np.asarray(d, float); n = len(d); se = d.std(ddof=1) / np.sqrt(n)
    t1 = (d.mean() + delta) / se; t2 = (d.mean() - delta) / se
    p1 = 1 - stats.t.cdf(t1, n - 1); p2 = stats.t.cdf(t2, n - 1)
    return float(max(p1, p2))


# valores criticos q_alpha de Nemenyi (alpha=0.05), k = numero de modelos
NEMENYI_Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850}


def friedman_nemenyi(M, names):
    """M: (n_bloques, k) metricas (menor es mejor). Devuelve estadistico, p, rangos medios, CD."""
    M = np.asarray(M, float); n, k = M.shape
    chi, p = stats.friedmanchisquare(*[M[:, j] for j in range(k)])
    ranks = np.apply_along_axis(stats.rankdata, 1, M).mean(0)
    cd = NEMENYI_Q05[k] * np.sqrt(k * (k + 1) / (6 * n))
    return dict(chi2=float(chi), p=float(p), mean_ranks=dict(zip(names, map(float, ranks))), CD=float(cd), n_blocks=n)


def variance_decomposition(M):
    """M: (n_semillas, k_modelos). Componentes de varianza (seed vs modelos) por ANOVA de una via."""
    M = np.asarray(M, float); n, k = M.shape
    within = M.var(axis=0, ddof=1).mean()           # variabilidad por semilla (dentro de modelo)
    between = M.mean(axis=0).var(ddof=1)            # variabilidad entre medias de modelo
    return dict(var_seed=float(within), var_between_models=float(between),
                frac_seed=float(within / (within + between)))
