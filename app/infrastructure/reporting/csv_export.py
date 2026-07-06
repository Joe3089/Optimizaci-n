# app/infrastructure/reporting/csv_export.py
# Exportación a CSV normalizados + script SQL — extraído de
# InterfazOptimizacion.export_csv_db (app/ui/main_window.py), Fase 4c.
from __future__ import annotations

import csv
import datetime
import math
import os
import re

import numpy as np

from app.application import report_content


def write_csv_db(folder: str, session: list, history_log: list) -> None:
    """
    Escribe 3 CSV normalizados + script SQL compatibles con:
    PostgreSQL, MySQL, SQLite, MariaDB, SQL Server, Oracle,
    DBeaver, pgAdmin, DataGrip, TablePlus, etc.

    Archivos generados (en `folder`):
      reporte_sesion.csv       — una fila por método ejecutado
      reporte_iteraciones.csv  — una fila por iteración
      reporte_historial.csv    — historial de funciones de la sesión
      crear_tablas.sql         — script CREATE TABLE
      importar_datos.sql       — script de importación (COPY)
    """
    def _sql_col(name: str) -> str:
        """Nombre SQL-safe: minúsculas, sin tildes, solo a-z0-9_."""
        t = name.lower().strip()
        t = t.replace("á","a").replace("é","e").replace("í","i")\
             .replace("ó","o").replace("ú","u").replace("ñ","n")
        t = re.sub(r"[^a-z0-9_]", "_", t).strip("_")
        return t or "col"

    timestamp_export = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _num(v):
        """Devuelve float limpio como string, o None — nunca NaN/Inf/string vacío con comillas."""
        if v is None:
            return None
        if isinstance(v, np.ndarray):
            flat = v.flatten()
            if flat.size == 1:
                v = float(flat[0])
            else:
                return None
        try:
            f = float(v)
            if math.isnan(f) or math.isinf(f):
                return None
            return f"{f:.10g}"
        except Exception:
            return None

    def _txt(v):
        """Devuelve texto limpio o cadena vacía."""
        if v is None:
            return ""
        return str(v)

    def _csv_writer(path):
        """CSV con comillas solo en campos de texto, numéricos sin comillas — compatible PostgreSQL."""
        fh = open(path, "w", newline="", encoding="utf-8")
        w = csv.writer(fh, delimiter=",",
                       quoting=csv.QUOTE_MINIMAL,
                       lineterminator="\r\n")
        return fh, w

    # ═══════════════════════════════════════════════════════════════════
    # TABLA 1 — reporte_sesion
    # ═══════════════════════════════════════════════════════════════════
    p_sesion = os.path.join(folder, "reporte_sesion.csv")
    fh1, w1 = _csv_writer(p_sesion)
    w1.writerow([
        "metodo_id", "metodo", "funcion_fx", "variables",
        "intervalo_min", "intervalo_max", "punto_inicial",
        "total_iteraciones", "x_optimo", "f_optimo",
        "tipo_estudio", "exportado_en"
    ])
    for idx, entry in enumerate(session, 1):
        hist  = entry.get("hist") or []
        vars_ = entry.get("vars", [])
        x0    = entry.get("x0", [])
        x_opt = f_opt = None
        if hist:
            last = hist[-1]
            for kx in ("x_k","x","lambda_k","mu_k","x1"):
                if kx in last:
                    x_opt = _num(last[kx]); break
            for kf in ("f_k","f(x)","f_new","f_lambda","f_mu","fx"):
                if kf in last:
                    f_opt = _num(last[kf]); break
        x0_str = ";".join(str(v) for v in x0) if x0 else ""
        w1.writerow([
            idx,
            _txt(entry["metodo"]),
            _txt(entry["fx"]),
            _txt(";".join(vars_)),
            _num(entry.get("lo")),
            _num(entry.get("hi")),
            x0_str,
            len(hist),
            x_opt,
            f_opt,
            _txt(report_content.classify_study(entry["metodo"])),
            timestamp_export,
        ])
    fh1.close()

    # ═══════════════════════════════════════════════════════════════════
    # TABLA 2 — reporte_iteraciones
    # ═══════════════════════════════════════════════════════════════════
    # Recolectar columnas únicas de todos los registros de todas las iteraciones
    orig_cols: list = []       # nombres originales
    for entry in session:
        for rec in (entry.get("hist") or []):
            for k in rec:
                if k not in orig_cols:
                    orig_cols.append(k)

    sql_cols = [_sql_col(k) for k in orig_cols]

    p_iter = os.path.join(folder, "reporte_iteraciones.csv")
    fh2, w2 = _csv_writer(p_iter)
    w2.writerow(["metodo_id", "metodo", "funcion_fx", "iteracion"] + sql_cols)
    for idx, entry in enumerate(session, 1):
        for i, rec in enumerate(entry.get("hist") or []):
            row = [idx, _txt(entry["metodo"]), _txt(entry["fx"]), i]
            for ok in orig_cols:
                row.append(_num(rec.get(ok)))
            w2.writerow(row)
    fh2.close()

    # ═══════════════════════════════════════════════════════════════════
    # TABLA 3 — reporte_historial
    # ═══════════════════════════════════════════════════════════════════
    p_hist = os.path.join(folder, "reporte_historial.csv")
    fh3, w3 = _csv_writer(p_hist)
    w3.writerow(["hora","funcion_fx","metodo","iteraciones","x_optimo","f_optimo"])
    for i, e in enumerate(history_log, 1):
        w3.writerow([
            _txt(e.get("timestamp","")),
            _txt(e.get("fx","")),
            _txt(e.get("metodo","")),
            e.get("iters") or None,
            _num(e.get("xopt")),
            _num(e.get("fopt")),
        ])
    fh3.close()

    # ═══════════════════════════════════════════════════════════════════
    # SCRIPT SQL — CREATE TABLE
    # ═══════════════════════════════════════════════════════════════════
    p_sql = os.path.join(folder, "crear_tablas.sql")
    folder_sql = folder.replace("\\", "/")
    iter_cols_ddl = "\n".join(
        f"    {c:<24} DOUBLE PRECISION," for c in sql_cols
    )
    with open(p_sql, "w", encoding="utf-8") as fs:
        fs.write(f"-- Generado por Optimizador de Funciones  |  {timestamp_export}\n")
        fs.write("-- PASO 1: Ejecuta este archivo en pgAdmin Query Tool para crear las tablas.\n")
        fs.write("-- PASO 2: Usa importar_datos.sql para cargar los CSV.\n\n")
        fs.write(
            "CREATE TABLE IF NOT EXISTS reporte_sesion (\n"
            "    metodo_id         SERIAL          PRIMARY KEY,\n"
            "    metodo            VARCHAR(120),\n"
            "    funcion_fx        TEXT,\n"
            "    variables         TEXT,\n"
            "    intervalo_min     DOUBLE PRECISION,\n"
            "    intervalo_max     DOUBLE PRECISION,\n"
            "    punto_inicial     TEXT,\n"
            "    total_iteraciones INTEGER,\n"
            "    x_optimo          DOUBLE PRECISION,\n"
            "    f_optimo          DOUBLE PRECISION,\n"
            "    tipo_estudio      VARCHAR(120),\n"
            "    exportado_en      TIMESTAMP\n"
            ");\n\n"
        )
        fs.write(
            "CREATE TABLE IF NOT EXISTS reporte_iteraciones (\n"
            "    id                SERIAL          PRIMARY KEY,\n"
            "    metodo_id         INTEGER         REFERENCES reporte_sesion(metodo_id),\n"
            "    metodo            VARCHAR(120),\n"
            "    funcion_fx        TEXT,\n"
            "    iteracion         INTEGER,\n"
        )
        fs.write(iter_cols_ddl + "\n")
        fs.write(
            "    _dummy            BOOLEAN DEFAULT NULL\n"
            ");\n\n"
        )
        fs.write(
            "CREATE TABLE IF NOT EXISTS reporte_historial (\n"
            "    id                SERIAL          PRIMARY KEY,\n"
            "    hora              VARCHAR(20),\n"
            "    funcion_fx        TEXT,\n"
            "    metodo            VARCHAR(120),\n"
            "    iteraciones       INTEGER,\n"
            "    x_optimo          DOUBLE PRECISION,\n"
            "    f_optimo          DOUBLE PRECISION\n"
            ");\n"
        )

    # SCRIPT SEPARADO — solo importación (COPY server-side para pgAdmin)
    p_import = os.path.join(folder, "importar_datos.sql")
    with open(p_import, "w", encoding="utf-8") as fi:
        fi.write(f"-- Importar datos  |  {timestamp_export}\n")
        fi.write("-- Ejecuta cada linea POR SEPARADO en pgAdmin Query Tool.\n")
        fi.write("-- El servidor PostgreSQL debe poder leer la ruta indicada.\n")
        fi.write("-- Si usas PostgreSQL local en Windows, la ruta debe ser con barras /\n\n")
        # Columnas explícitas para reporte_sesion (excluye 'id' SERIAL)
        sesion_cols = (
            "metodo_id, metodo, funcion_fx, variables, intervalo_min, "
            "intervalo_max, punto_inicial, total_iteraciones, x_optimo, "
            "f_optimo, tipo_estudio, exportado_en"
        )
        # Columnas explícitas para reporte_iteraciones (excluye 'id' SERIAL y '_dummy')
        iter_copy_cols = ", ".join(
            ["metodo_id", "metodo", "funcion_fx", "iteracion"] + sql_cols
        )
        fi.write(
            f"COPY reporte_sesion({sesion_cols})\n"
            f"  FROM '{folder_sql}/reporte_sesion.csv'\n"
            f"  WITH (FORMAT CSV, HEADER TRUE, DELIMITER ',', ENCODING 'UTF8');\n\n"
        )
        fi.write(
            f"COPY reporte_iteraciones({iter_copy_cols})\n"
            f"  FROM '{folder_sql}/reporte_iteraciones.csv'\n"
            f"  WITH (FORMAT CSV, HEADER TRUE, DELIMITER ',', ENCODING 'UTF8');\n\n"
        )
        fi.write(
            f"COPY reporte_historial(hora, funcion_fx, metodo, iteraciones, x_optimo, f_optimo)\n"
            f"  FROM '{folder_sql}/reporte_historial.csv'\n"
            f"  WITH (FORMAT CSV, HEADER TRUE, DELIMITER ',', ENCODING 'UTF8');\n\n"
            "-- ALTERNATIVA: si prefieres usar la interfaz grafica de pgAdmin:\n"
            "--   Click derecho en la tabla > Import/Export Data\n"
            "--   Format: csv  |  Header: ON  |  Delimiter: ,  |  Encoding: UTF8\n"
        )
