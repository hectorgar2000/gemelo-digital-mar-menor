"""Funciones puras de deriva (sin modelos ni ficheros): se pueden probar con datos sintéticos."""
import numpy as np
import pandas as pd

from gemelo.procesado.features import _doy


def psi(referencia, actual, bins: int = 10, eps: float = 1e-4) -> float:
    """Population Stability Index: cuánto se ha desplazado la distribución `actual` respecto a `referencia`.
    Tramos = deciles de la referencia. Regla habitual: < 0,10 estable, 0,10-0,25 moderado, > 0,25 fuerte."""
    ref = np.asarray(referencia, dtype=float)
    act = np.asarray(actual, dtype=float)
    ref, act = ref[np.isfinite(ref)], act[np.isfinite(act)]
    if len(ref) < bins or len(act) == 0:
        return float("nan")
    cortes = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)[1:-1]))
    p_ref = np.bincount(np.searchsorted(cortes, ref, side="right"), minlength=len(cortes) + 1) / len(ref)
    p_act = np.bincount(np.searchsorted(cortes, act, side="right"), minlength=len(cortes) + 1) / len(act)
    p_ref, p_act = np.clip(p_ref, eps, None), np.clip(p_act, eps, None)
    return float(np.sum((p_act - p_ref) * np.log(p_act / p_ref)))


def mascara_estacional(indice_ref: pd.DatetimeIndex, fechas_ventana: pd.DatetimeIndex, margen: int) -> np.ndarray:
    """True para las fechas de referencia cuyo día del año cae en el de la ventana (± margen, circular).
    Así un verano se compara con veranos, no con la media de todo el año."""
    doy_ref = _doy(indice_ref)
    objetivo = np.zeros(366, dtype=bool)
    for d in np.unique(_doy(fechas_ventana)):
        for k in range(-margen, margen + 1):
            objetivo[((d - 1 + k) % 365) + 1] = True
    return objetivo[doy_ref]


def fraccion_fuera_de_rango(X: pd.DataFrame, minimo: pd.Series, maximo: pd.Series) -> float:
    """Fracción de filas con ALGUNA variable fuera del rango [mínimo, máximo] visto en entrenamiento."""
    fuera = (X.lt(minimo[X.columns], axis=1) | X.gt(maximo[X.columns], axis=1)).any(axis=1)
    return float(fuera.mean()) if len(X) else float("nan")


def intervalos_conformales(residuos: pd.Series, h: int, ventana: int, min_residuos: int,
                           q_fijos: tuple[float, float]) -> pd.DataFrame:
    """Para cada día t, los cuantiles 5 % y 95 % de los residuos de los días [t-ventana-h, t-h] (causal), igual
    que Gemelo.residuos_recientes. Si hay menos de `min_residuos`, los cuantiles fijos de validación."""
    r = residuos.asfreq("D")
    rol = r.rolling(f"{ventana + 1}D", min_periods=min_residuos)
    lo, hi, n = rol.quantile(0.05), rol.quantile(0.95), rol.count()
    # el dato de t usa la ventana que termina en t-h: desplazar h días
    lo, hi, n = (s.shift(h, freq="D").reindex(r.index) for s in (lo, hi, n))
    usar_fijos = n.isna() | (n < min_residuos)
    return pd.DataFrame({"lo": lo.where(~usar_fijos, q_fijos[0]), "hi": hi.where(~usar_fijos, q_fijos[1])})


def cuantiles_conformales(dias_res: np.ndarray, doy_res: np.ndarray, valores: np.ndarray, t: pd.Timestamp, h: int,
                          cfg_int: dict, q_fijos: tuple[float, float]) -> tuple[float, float, str, np.ndarray]:
    """Cuantiles 5 % / 95 % de los residuos para el día t, en cascada (todo causal: solo residuos de días <= t-h):
      1. estacional: misma época (± margen días) de los últimos `anios` años, si hay >= min_residuos;
      2. móvil: los últimos 365 días (el método anterior), si hay >= min_residuos;
      3. fijo: cuantiles de validación guardados al entrenar.
    Los errores de verano son mucho mayores que los de invierno: mezclarlos (método 2) da intervalos estrechos
    en verano y anchos en invierno. Por eso la opción preferida es la estacional.
    `dias_res` = fechas de los residuos en días enteros (ns // día), `doy_res` = día del año."""
    t_dia = t.value // 86_400_000_000_000
    fin = t_dia - h
    doy_t = int(_doy(pd.DatetimeIndex([t]))[0])
    dist = np.abs(doy_res - doy_t)
    dist = np.minimum(dist, 365 - dist)
    mn = cfg_int["min_residuos"]
    sel = (dias_res <= fin) & (dias_res >= fin - 365 * cfg_int["anios"]) & (dist <= cfg_int["margen_dias"])
    if sel.sum() >= mn:
        v = valores[sel]
        return float(np.quantile(v, 0.05)), float(np.quantile(v, 0.95)), "conformal_estacional", v
    sel = (dias_res <= fin) & (dias_res >= fin - 365)
    if sel.sum() >= mn:
        v = valores[sel]
        return float(np.quantile(v, 0.05)), float(np.quantile(v, 0.95)), "conformal_movil_365d", v
    return q_fijos[0], q_fijos[1], "cuantiles_validacion", np.array([q_fijos[0], q_fijos[1]])


def intervalos_cascada(residuos: pd.Series, fechas, h: int, cfg_int: dict, q_fijos) -> pd.DataFrame:
    """cuantiles_conformales para muchas fechas (lo usa el monitor; la API llama a la misma función)."""
    r = residuos.dropna()
    dias = r.index.values.astype("datetime64[D]").astype(np.int64)
    doy = _doy(r.index)
    filas = [cuantiles_conformales(dias, doy, r.values, t, h, cfg_int, q_fijos)[:3] for t in fechas]
    return pd.DataFrame(filas, index=fechas, columns=["lo", "hi", "metodo"])


def interval_score(y, pred, lo, hi, alfa: float = 0.10) -> float:
    """Interval score de Winkler (Gneiting y Raftery, 2007): ancho + (2/alfa)·distancia de los fallos.
    Premia intervalos estrechos que aciertan y castiga cada fallo según lo lejos que cae. Menor = mejor."""
    y, pred, lo, hi = (np.asarray(a, dtype=float) for a in (y, pred, lo, hi))
    inf, sup = pred + lo, pred + hi
    return float(np.mean((sup - inf) + (2 / alfa) * (inf - y) * (y < inf) + (2 / alfa) * (y - sup) * (y > sup)))


def cobertura(y, pred, lo, hi) -> float:
    y, pred, lo, hi = (np.asarray(a, dtype=float) for a in (y, pred, lo, hi))
    return float(np.mean((y >= pred + lo) & (y <= pred + hi)))


def estado(skill: float, cob: float, psi_max: float, fuera: float, cfg_mon: dict) -> tuple[str, list[str]]:
    """OK / AVISO / DERIVA y los motivos.
    - DERIVA: ha empeorado el rendimiento (no bate a la persistencia o el intervalo no cubre): hay que actuar.
    - AVISO: han cambiado las entradas (PSI alto o fuera de rango) aunque el rendimiento aguante."""
    motivos = []
    if np.isfinite(skill) and skill < cfg_mon["skill_min"]:
        motivos.append(f"no bate a la persistencia (skill {skill:+.1%})")
    if np.isfinite(cob) and cob < cfg_mon["cobertura_min"]:
        motivos.append(f"el intervalo del 90 % solo cubre el {cob:.0%}")
    if motivos:
        return "DERIVA", motivos
    if np.isfinite(psi_max) and psi_max > cfg_mon["psi_deriva"]:
        motivos.append(f"entradas desplazadas (PSI máx. {psi_max:.2f})")
    if np.isfinite(fuera) and fuera > cfg_mon["max_fuera_rango"]:
        motivos.append(f"{fuera:.0%} de los días con variables fuera del rango de entrenamiento")
    return ("AVISO" if motivos else "OK"), motivos
