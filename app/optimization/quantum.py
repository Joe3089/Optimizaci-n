# app/optimization/quantum.py
# Optimización cuántica sobre un simulador de vector de estado en NumPy. No se
# conecta a hardware cuántico: el costo crece como 2^n, por lo que cada método
# limita el número de qubits.
from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Sequence

import numpy as np
from scipy.optimize import minimize

from app.optimization.combinatorial import Edge, bits, check_graph, cut_values, graph_plot

H = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)
PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def apply_1q(psi: np.ndarray, U: np.ndarray, q: int) -> np.ndarray:
    idx = np.arange(len(psi))
    i0 = idx[(idx >> q) & 1 == 0]
    i1 = i0 | (1 << q)
    a, b = psi[i0].copy(), psi[i1].copy()
    psi[i0] = U[0, 0] * a + U[0, 1] * b
    psi[i1] = U[1, 0] * a + U[1, 1] * b
    return psi


def apply_cnot(psi: np.ndarray, c: int, t: int) -> np.ndarray:
    idx = np.arange(len(psi))
    i0 = idx[((idx >> c) & 1 == 1) & ((idx >> t) & 1 == 0)]
    i1 = i0 | (1 << t)
    psi[i0], psi[i1] = psi[i1].copy(), psi[i0].copy()
    return psi


def _plus(n: int) -> np.ndarray:
    return np.full(1 << n, 1 / math.sqrt(1 << n), dtype=complex)


def _rx_all(psi: np.ndarray, beta: float, n: int) -> np.ndarray:
    """Aplica exp(−iβ·ΣX) con un intercambio de amplitudes por qubit."""
    idx = np.arange(len(psi))
    c, s = math.cos(beta), -1j * math.sin(beta)
    for q in range(n):
        psi = c * psi + s * psi[idx ^ (1 << q)]
    return psi


def _bar(fig, labels, values, title, ylabel="Probabilidad"):
    ax = fig.add_subplot(111)
    ax.bar(labels, values, color="#4a9eff")
    ax.set_title(title); ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45)


# ── Conceptos ────────────────────────────────────────────────────────────────
def paradigm_shift(qubits: Sequence[int]) -> Dict[str, Any]:
    if not qubits or any(q < 1 or q > 60 or int(q) != q for q in qubits):
        raise ValueError("Los qubits deben ser enteros entre 1 y 60.")
    hist: List[Dict[str, Any]] = [
        {"Aspecto": "Unidad de información", "Clásico": "Bit: 0 o 1", "Cuántico": "Qubit: α|0⟩ + β|1⟩"},
        {"Aspecto": "Estado de n unidades", "Clásico": "Una de 2ⁿ cadenas", "Cuántico": "2ⁿ amplitudes complejas"},
        {"Aspecto": "Operaciones", "Clásico": "Compuertas lógicas", "Cuántico": "Matrices unitarias reversibles"},
        {"Aspecto": "Lectura", "Clásico": "No altera el estado", "Cuántico": "Colapsa a un resultado probabilístico"},
        {"Aspecto": "Estrategia de búsqueda", "Clásico": "Probar candidatos", "Cuántico": "Interferencia que amplifica soluciones"},
        {"Aspecto": "Ventaja demostrada", "Clásico": "Referencia", "Cuántico": "Grover: cuadrática; no resuelve NP-duros en tiempo polinomial conocido"},
    ]
    qs = sorted(int(q) for q in qubits)
    cl, qu = [], []
    for n in qs:
        N = 2.0 ** n
        c, g = N / 2, math.pi / 4 * math.sqrt(N)
        cl.append(c); qu.append(g)
        hist.append({"Aspecto": f"Buscar en 2^{n} elementos", "Clásico": f"{c:.4g} consultas en promedio",
                     "Cuántico": f"{g:.4g} consultas (Grover)"})

    def plot(fig):
        ax = fig.add_subplot(111)
        ax.semilogy(qs, cl, "o-", color="#ff6b35", label="Clásico (N/2)")
        ax.semilogy(qs, qu, "s-", color="#4a9eff", label="Grover (π/4·√N)")
        ax.set_xlabel("Qubits n (N = 2ⁿ)"); ax.set_ylabel("Consultas")
        ax.set_title("Costo de búsqueda no estructurada"); ax.legend()

    return {"hist": hist, "plot": plot,
            "resumen": f"Con {qs[-1]} qubits: clásico {cl[-1]:.3g} consultas, Grover {qu[-1]:.3g}"}


def phenomenon(kind: str, shots: int, seed: int) -> Dict[str, Any]:
    if not 1 <= shots <= 1_000_000:
        raise ValueError("Las mediciones deben estar entre 1 y 1 000 000.")
    rng = np.random.default_rng(seed)
    if kind == "Superposición":
        n, psi = 1, np.array([1, 0], dtype=complex)
        psi = apply_1q(psi, H, 0)
        expl = "H|0⟩: el qubit tiene amplitud en |0⟩ y |1⟩ a la vez"
    elif kind == "Entrelazamiento":
        n, psi = 2, np.array([1, 0, 0, 0], dtype=complex)
        psi = apply_cnot(apply_1q(psi, H, 0), 0, 1)
        expl = "Estado de Bell: los dos qubits siempre coinciden al medirlos"
    elif kind == "Interferencia":
        n, psi = 1, np.array([1, 0], dtype=complex)
        psi = apply_1q(apply_1q(psi, H, 0), H, 0)
        expl = "H·H|0⟩ = |0⟩: las amplitudes de |1⟩ se cancelan"
    else:
        raise ValueError(f"Fenómeno desconocido: {kind}.")
    probs = np.abs(psi) ** 2
    probs = probs / probs.sum()
    counts = rng.multinomial(shots, probs)
    labels = [f"|{bits(i, n)}⟩" for i in range(1 << n)]
    hist = [{"Estado": labels[i], "Amplitud": f"{psi[i].real:.4f}{psi[i].imag:+.4f}i",
             "Probabilidad teórica": float(probs[i]), "Mediciones": int(counts[i]),
             "Frecuencia": float(counts[i] / shots)} for i in range(1 << n)]
    hist.append({"Estado": "Explicación", "Amplitud": expl})

    def plot(fig):
        _bar(fig, labels, counts / shots, f"{kind}: {shots} mediciones", "Frecuencia")

    return {"probs": probs.tolist(), "counts": counts.tolist(), "hist": hist, "plot": plot,
            "resumen": expl}


def bit_vs_qubit(theta_deg: float, phi_deg: float, shots: int, seed: int) -> Dict[str, Any]:
    if not 0 <= theta_deg <= 180:
        raise ValueError("θ debe estar entre 0 y 180 grados.")
    if not 1 <= shots <= 1_000_000:
        raise ValueError("Las mediciones deben estar entre 1 y 1 000 000.")
    th, ph = math.radians(theta_deg), math.radians(phi_deg)
    a, b = math.cos(th / 2), complex(math.cos(ph), math.sin(ph)) * math.sin(th / 2)
    p0, p1 = a ** 2, abs(b) ** 2
    rng = np.random.default_rng(seed)
    ones = int(rng.binomial(shots, min(p1, 1.0)))
    bx, by, bz = math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th)
    hist = [{"Concepto": k, "Valor": v} for k, v in [
        ("Bit clásico", "Valor definido: 0 o 1"),
        ("Qubit", f"{a:.4f}|0⟩ + ({b.real:.4f}{b.imag:+.4f}i)|1⟩"),
        ("P(medir 0)", p0), ("P(medir 1)", p1),
        ("Bloch x", bx), ("Bloch y", by), ("Bloch z", bz),
        ("Frecuencia observada de 1", ones / shots),
        ("Amplitudes para 10 qubits", 2 ** 10), ("Amplitudes para 50 qubits", f"{2.0 ** 50:.4g}"),
    ]]

    def plot(fig):
        ax1, ax2 = fig.subplots(1, 2)
        ax1.bar(["P(0)", "P(1)"], [p0, p1], color=["#4a9eff", "#ffd700"])
        ax1.set_ylim(0, 1); ax1.set_title("Probabilidades de medición")
        t = np.linspace(0, 2 * np.pi, 200)
        ax2.plot(np.cos(t), np.sin(t), color="#35507f")
        ax2.plot([0, bx], [0, bz], "o-", color="#ffd700")
        ax2.set_aspect("equal"); ax2.set_xlabel("x"); ax2.set_ylabel("z")
        ax2.set_title("Esfera de Bloch (plano x-z)")

    return {"p0": p0, "p1": p1, "bloch": (bx, by, bz), "ones": ones, "hist": hist, "plot": plot,
            "resumen": f"P(0) = {p0:.4f}, P(1) = {p1:.4f}"}


# ── Algoritmos ───────────────────────────────────────────────────────────────
def _top_rows(probs, cuts, n, k=8):
    return [{"Iter": "Muestra", "Cadena": bits(int(i), n), "Probabilidad": float(probs[i]),
             "Corte": float(cuts[i])} for i in np.argsort(-probs)[:k]]


def qaoa(n: int, edges: Sequence[Edge], p: int, seed: int) -> Dict[str, Any]:
    check_graph(n, edges, 12)
    if not 1 <= p <= 6:
        raise ValueError("El número de capas p debe estar entre 1 y 6.")
    cuts = cut_values(n, edges)
    best = float(cuts.max())
    if best <= 0:
        raise ValueError("Las aristas deben tener peso positivo.")

    def state(params):
        psi = _plus(n)
        for g, b in zip(params[:p], params[p:]):
            psi = psi * np.exp(-1j * g * cuts)
            psi = _rx_all(psi, b, n)
        return psi

    def objective(params):
        return -float(np.abs(state(params)) ** 2 @ cuts)

    rng = np.random.default_rng(seed)
    best_res, trace = None, []
    for _ in range(12):
        x0 = np.concatenate([rng.uniform(0, np.pi, p), rng.uniform(0, np.pi / 2, p)])
        run: List[float] = []
        res = minimize(objective, x0, method="COBYLA", callback=lambda xk: run.append(-objective(xk)),
                       options={"maxiter": 250 * p, "rhobeg": 0.4})
        if best_res is None or res.fun < best_res.fun:
            best_res, trace = res, run

    probs = np.abs(state(best_res.x)) ** 2
    exp_cut = -float(best_res.fun)
    p_opt = float(probs[np.isclose(cuts, best)].sum())
    top = int(np.argmax(probs))
    step = max(1, len(trace) // 25)
    hist: List[Dict[str, Any]] = [
        {"Iter": i + 1, "⟨C⟩": float(v), "Razón ⟨C⟩/óptimo": float(v) / best}
        for i, v in list(enumerate(trace))[::step]]
    hist.append({"Iter": "Final", "⟨C⟩": exp_cut, "Razón ⟨C⟩/óptimo": exp_cut / best,
                 "Probabilidad": p_opt, "Cadena": "P(óptimo)"})
    hist += _top_rows(probs, cuts, n)
    side = [(top >> q) & 1 for q in range(n)]

    def plot(fig):
        graph_plot(fig, n, edges, side, f"QAOA p={p}: ⟨C⟩ = {exp_cut:.3f} de {best:g}")

    return {"expected": exp_cut, "best": best, "ratio": exp_cut / best, "p_opt": p_opt,
            "top_cut": float(cuts[top]), "hist": hist, "plot": plot,
            "resumen": f"⟨C⟩ = {exp_cut:.4f}, razón = {exp_cut / best:.3f}, "
                       f"P(óptimo) = {p_opt:.3f}, mejor cadena {bits(top, n)}"}


def parse_hamiltonian(text: str):
    terms = []
    for chunk in filter(str.strip, re.split(r"[;\n]", text)):
        parts = chunk.split()
        if len(parts) != 2:
            raise ValueError(f"Término inválido: «{chunk}». Formato: -1.0 ZZ")
        try:
            coef = float(parts[0])
        except ValueError as ex:
            raise ValueError(f"Coeficiente inválido en «{chunk}».") from ex
        word = parts[1].upper()
        if not set(word) <= set("IXYZ"):
            raise ValueError(f"Operador inválido en «{chunk}»: solo I, X, Y, Z.")
        terms.append((coef, word))
    if not terms:
        raise ValueError("Ingresa al menos un término del Hamiltoniano.")
    if len({len(w) for _, w in terms}) != 1:
        raise ValueError("Todos los términos deben tener el mismo número de qubits.")
    return terms


def vqe(text: str, layers: int, seed: int) -> Dict[str, Any]:
    terms = parse_hamiltonian(text)
    n = len(terms[0][1])
    if n > 8:
        raise ValueError("VQE admite hasta 8 qubits.")
    if not 1 <= layers <= 6:
        raise ValueError("Las capas deben estar entre 1 y 6.")
    Ham = np.zeros((1 << n, 1 << n), dtype=complex)
    for coef, word in terms:
        M = np.array([[1]], dtype=complex)
        for ch in reversed(word):  # el qubit 0 es el bit menos significativo
            M = np.kron(M, PAULI[ch])
        Ham += coef * M
    exact = float(np.linalg.eigvalsh(Ham)[0])

    def ry(t):
        return np.array([[math.cos(t / 2), -math.sin(t / 2)],
                         [math.sin(t / 2), math.cos(t / 2)]], dtype=complex)

    def rz(t):
        return np.diag([np.exp(-0.5j * t), np.exp(0.5j * t)])

    size = 2 * n * (layers + 1)

    def state(params):
        psi = np.zeros(1 << n, dtype=complex)
        psi[0] = 1
        pos = 0
        for layer in range(layers + 1):
            for q in range(n):
                psi = apply_1q(psi, ry(params[pos]), q)
                psi = apply_1q(psi, rz(params[pos + 1]), q)
                pos += 2
            if layer < layers:
                for q in range(n - 1):
                    psi = apply_cnot(psi, q, q + 1)
        return psi

    def energy(params):
        psi = state(params)
        return float(np.real(psi.conj() @ Ham @ psi))

    rng = np.random.default_rng(seed)
    best, trace = None, []
    for _ in range(6):
        run: List[float] = []
        res = minimize(energy, rng.uniform(-np.pi, np.pi, size), method="L-BFGS-B",
                       callback=lambda xk: run.append(energy(xk)), options={"maxiter": 400})
        if best is None or res.fun < best.fun:
            best, trace = res, run
    e = float(best.fun)
    step = max(1, len(trace) // 25)
    hist = [{"Iter": i + 1, "Energía": float(v), "Error absoluto": abs(float(v) - exact)}
            for i, v in list(enumerate(trace))[::step]]
    hist.append({"Iter": "Final", "Energía": e, "Error absoluto": abs(e - exact),
                 "Referencia exacta": exact})

    def plot(fig):
        ax = fig.add_subplot(111)
        ax.plot(range(1, len(trace) + 1), trace, color="#4a9eff", label="VQE")
        ax.axhline(exact, color="#ffd700", linestyle="--", label="Exacta")
        ax.set_xlabel("Iteración"); ax.set_ylabel("Energía"); ax.set_title("VQE"); ax.legend()

    return {"energy": e, "exact": exact, "hist": hist, "plot": plot,
            "resumen": f"Energía VQE = {e:.6f}, exacta = {exact:.6f}"}


def grover(n: int, marked: Sequence[str]) -> Dict[str, Any]:
    if not 1 <= n <= 16:
        raise ValueError("Grover admite entre 1 y 16 qubits.")
    idx = set()
    for s in marked:
        if len(s) != n or not set(s) <= {"0", "1"}:
            raise ValueError(f"Estado marcado inválido: «{s}». Usa cadenas de {n} bits.")
        idx.add(sum(int(ch) << q for q, ch in enumerate(s)))
    if not idx:
        raise ValueError("Indica al menos un estado marcado.")
    N, M = 1 << n, len(idx)
    if M >= N:
        raise ValueError("No puede marcarse todo el espacio de búsqueda.")
    mask = np.zeros(N, dtype=bool)
    mask[list(idx)] = True
    k_opt = max(1, math.floor(math.pi / 4 * math.sqrt(N / M)))
    theta = math.asin(math.sqrt(M / N))
    psi = _plus(n)
    hist, series = [], []
    for k in range(0, k_opt + 3):
        p = float((np.abs(psi[mask]) ** 2).sum())
        series.append(p)
        hist.append({"Iteración k": k, "P(éxito) simulada": p,
                     "P(éxito) teórica": math.sin((2 * k + 1) * theta) ** 2,
                     "Nota": "Número óptimo de iteraciones" if k == k_opt else ""})
        psi[mask] *= -1
        psi = 2 * psi.mean() - psi
    p_opt = series[k_opt]

    def plot(fig):
        ax = fig.add_subplot(111)
        ax.plot(range(len(series)), series, "o-", color="#4a9eff")
        ax.axvline(k_opt, color="#ffd700", linestyle="--", label=f"k óptimo = {k_opt}")
        ax.set_xlabel("Iteraciones de Grover"); ax.set_ylabel("P(éxito)")
        ax.set_title("Amplificación de amplitud"); ax.legend()

    return {"k_opt": k_opt, "p_success": p_opt, "hist": hist, "plot": plot,
            "resumen": f"{k_opt} iteraciones: P(éxito) = {p_opt:.4f}"}


def quantum_annealing(n: int, edges: Sequence[Edge], total_time: float, steps: int) -> Dict[str, Any]:
    check_graph(n, edges, 12)
    if total_time <= 0 or not 10 <= steps <= 20000:
        raise ValueError("El tiempo debe ser positivo y los pasos estar entre 10 y 20 000.")
    cuts = cut_values(n, edges)
    best = float(cuts.max())
    mask = np.isclose(cuts, best)
    psi = _plus(n)
    dt = total_time / steps
    hist, ss, ps = [], [], []
    every = max(1, steps // 25)
    for k in range(steps + 1):
        s = k / steps
        if k % every == 0 or k == steps:
            probs = np.abs(psi) ** 2
            p = float(probs[mask].sum())
            ss.append(s); ps.append(p)
            hist.append({"s": s, "⟨C⟩": float(probs @ cuts), "P(óptimo)": p})
        if k < steps:
            sm = (k + 0.5) / steps
            # H(s) = −(1−s)·ΣX − s·C, con división de Trotter simétrica.
            psi = _rx_all(psi, -0.5 * dt * (1 - sm), n)
            psi = psi * np.exp(1j * dt * sm * cuts)
            psi = _rx_all(psi, -0.5 * dt * (1 - sm), n)
    probs = np.abs(psi) ** 2
    p_final = float(probs[mask].sum())
    top = int(np.argmax(probs))
    side = [(top >> q) & 1 for q in range(n)]

    def plot(fig):
        ax = fig.add_subplot(111)
        ax.plot(ss, ps, color="#4a9eff")
        ax.set_xlabel("s = t/T"); ax.set_ylabel("P(óptimo)")
        ax.set_title("Recocido cuántico simulado")

    return {"p_opt": p_final, "best": best, "side": side, "expected": float(probs @ cuts),
            "hist": hist, "plot": plot,
            "resumen": f"P(óptimo) = {p_final:.4f}, ⟨C⟩ = {float(probs @ cuts):.4f} de {best:g}"}
