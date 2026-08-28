"""Gera os dados sintéticos (parquet) + seed.json que populam a API.

Coerente com docs/data-schema.md e o mapeamento chamado→dados. Dados anonimizados,
sem PII. Roda com `python -m seed_data` (a partir de api/) — gera ../data/*.parquet.

Reprodutível: random.seed fixo; timestamps base fixos (offsets em dias a partir de
uma época de referência).
"""
from __future__ import annotations

import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

# Reprodutibilidade
RNG = random.Random(42)

# Época de referência: 2026-07-15 00:00 UTC (todos os timestamps são offsets a partir daqui)
EPOCH = datetime(2026, 7, 15, 0, 0, 0, tzinfo=timezone.utc)


def ts(days: float = 0.0, hours: float = 0.0) -> str:
    return (EPOCH + timedelta(days=days, hours=hours)).isoformat()


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# ---------------------------------------------------------------------------
# Empresas e usuários
# ---------------------------------------------------------------------------
COMPANIES = [
    {"id": "comp_forja_br", "name": "Forja Brasil", "segment": "metalurgia", "timezone": "America/Sao_Paulo"},
    {"id": "comp_aurora", "name": "Cervejaria Aurora", "segment": "alimentos", "timezone": "America/Sao_Paulo"},
    {"id": "comp_papel_sul", "name": "Papel Sul", "segment": "papel_celulose", "timezone": "America/Sao_Paulo"},
    {"id": "comp_mineracao_andes", "name": "Mineração Andes", "segment": "mineracao", "timezone": "America/Sao_Paulo"},
    {"id": "comp_petro_delta", "name": "Petro Delta", "segment": "oleo_gas", "timezone": "America/Sao_Paulo"},
    {"id": "comp_acme", "name": "Acme Auto Peças", "segment": "automotivo", "timezone": "America/Sao_Paulo"},
    {"id": "comp_cimento_vale", "name": "Cimento Vale", "segment": "cimento", "timezone": "America/Sao_Paulo"},
    {"id": "comp_texfil", "name": "Texfil", "segment": "textil", "timezone": "America/Sao_Paulo"},
]

USERS = [
    {"id": "usr_ana", "name": "Ana Mantovani", "role": "maintenance_manager", "permissions": '["read","action_high","escalate"]', "company_id": "comp_forja_br"},
    {"id": "usr_lucas", "name": "Lucas Pereira", "role": "mechanic", "permissions": '["read","action_low"]', "company_id": "comp_aurora"},
    {"id": "usr_marta", "name": "Marta Ribeiro", "role": "reliability_analyst", "permissions": '["read","action_low"]', "company_id": "comp_papel_sul"},
    {"id": "usr_helena", "name": "Helena Castro", "role": "maintenance_manager", "permissions": '["read","action_high","escalate"]', "company_id": "comp_papel_sul"},
    {"id": "usr_pedro", "name": "Pedro Alves", "role": "coordinator", "permissions": '["read","escalate"]', "company_id": "comp_mineracao_andes"},
    {"id": "usr_sofia", "name": "Sofia Nunes", "role": "reliability_analyst", "permissions": '["read","action_low"]', "company_id": "comp_petro_delta"},
    {"id": "usr_bruno", "name": "Bruno Dias", "role": "operator", "permissions": '["read"]', "company_id": "comp_acme"},
    {"id": "usr_carla", "name": "Carla Mendes", "role": "engineer", "permissions": '["read","action_high"]', "company_id": "comp_cimento_vale"},
    {"id": "usr_raul", "name": "Raul Souza", "role": "electrician", "permissions": '["read"]', "company_id": "comp_texfil"},
    {"id": "usr_gustavo", "name": "Gustavo Lima", "role": "mechanic", "permissions": '["read","action_low"]', "company_id": "comp_cimento_vale"},
]


# ---------------------------------------------------------------------------
# Ativos (config técnica sem ISO; specs de rolamento onde relevante)
# ---------------------------------------------------------------------------
ASSETS = [
    # Forja Brasil
    {"id": "asset_M101", "name": "Motor principal da forja", "company_id": "comp_forja_br", "criticality": "critical", "plant": "Planta 1", "line": "Forjamento", "parent_asset_id": None, "machine_type": "motor_induction", "rotation_rpm": 1780, "bearing_pn": "NU 310", "bpfo_hz": 142.3, "bpfi_hz": 218.1, "bsf_hz": 58.7, "ftf_hz": 11.9, "line_frequency_hz": 60, "sensor_status": "online"},
    {"id": "asset_M102", "name": "Motor CC antigo", "company_id": "comp_forja_br", "criticality": "medium", "plant": "Planta 1", "line": "Linha auxiliar", "parent_asset_id": None, "machine_type": "motor_dc", "rotation_rpm": 1200, "bearing_pn": None, "bpfo_hz": None, "bpfi_hz": None, "bsf_hz": None, "ftf_hz": None, "line_frequency_hz": None, "sensor_status": "online"},
    # Cervejaria Aurora
    {"id": "asset_B204", "name": "Bomba de mosto", "company_id": "comp_aurora", "criticality": "high", "plant": "Planta 1", "line": "Mosturação", "parent_asset_id": None, "machine_type": "pump", "rotation_rpm": 1750, "bearing_pn": "6309", "bpfo_hz": 107.4, "bpfi_hz": 169.6, "bsf_hz": 71.2, "ftf_hz": 7.1, "line_frequency_hz": None, "sensor_status": "online"},
    # Papel Sul
    {"id": "asset_V301", "name": "Ventilador de tiragem", "company_id": "comp_papel_sul", "criticality": "high", "plant": "Planta 2", "line": "Caldeira", "parent_asset_id": None, "machine_type": "fan", "rotation_rpm": 1180, "bearing_pn": "22220", "bpfo_hz": 83.6, "bpfi_hz": 121.4, "bsf_hz": 49.8, "ftf_hz": 9.7, "line_frequency_hz": None, "sensor_status": "degraded"},
    # Mineração Andes
    {"id": "asset_G501", "name": "Redutor da correia transportadora", "company_id": "comp_mineracao_andes", "criticality": "critical", "plant": "Mina Norte", "line": "Transporte", "parent_asset_id": None, "machine_type": "gearbox", "rotation_rpm": 48, "bearing_pn": None, "bpfo_hz": None, "bpfi_hz": None, "bsf_hz": None, "ftf_hz": None, "line_frequency_hz": None, "sensor_status": "offline"},
    # Petro Delta
    {"id": "asset_C710", "name": "Compressor de gás", "company_id": "comp_petro_delta", "criticality": "critical", "plant": "Plataforma A", "line": "Compressão", "parent_asset_id": None, "machine_type": "compressor", "rotation_rpm": 2950, "bearing_pn": "7321", "bpfo_hz": 198.2, "bpfi_hz": 287.5, "bsf_hz": 102.1, "ftf_hz": 16.4, "line_frequency_hz": None, "sensor_status": "online"},
    # Acme Auto Peças
    {"id": "asset_S420", "name": "Spindle de usinagem", "company_id": "comp_acme", "criticality": "high", "plant": "Planta 1", "line": "Usinagem", "parent_asset_id": None, "machine_type": "spindle", "rotation_rpm": 12000, "bearing_pn": "7008", "bpfo_hz": 721.4, "bpfi_hz": 1098.2, "bsf_hz": 312.7, "ftf_hz": 48.1, "line_frequency_hz": None, "sensor_status": "online"},
    # Cimento Vale
    {"id": "asset_M205", "name": "Moinho de cimento", "company_id": "comp_cimento_vale", "criticality": "critical", "plant": "Planta 1", "line": "Moagem", "parent_asset_id": None, "machine_type": "mill", "rotation_rpm": 240, "bearing_pn": "23236", "bpfo_hz": 41.7, "bpfi_hz": 60.3, "bsf_hz": 24.8, "ftf_hz": 4.9, "line_frequency_hz": None, "sensor_status": "online"},
    {"id": "asset_M208", "name": "Motor do moinho (novo)", "company_id": "comp_cimento_vale", "criticality": "high", "plant": "Planta 1", "line": "Moagem", "parent_asset_id": None, "machine_type": "motor_induction", "rotation_rpm": 1780, "bearing_pn": "6310", "bpfo_hz": 109.2, "bpfi_hz": 167.8, "bsf_hz": 70.4, "ftf_hz": 7.3, "line_frequency_hz": 60, "sensor_status": "online"},
    # Texfil
    {"id": "asset_M605", "name": "Motor de alta velocidade", "company_id": "comp_texfil", "criticality": "high", "plant": "Planta 1", "line": "Tecelagem", "parent_asset_id": None, "machine_type": "motor_induction", "rotation_rpm": 3550, "bearing_pn": "6205", "bpfo_hz": 173.1, "bpfi_hz": 266.9, "bsf_hz": 89.7, "ftf_hz": 14.2, "line_frequency_hz": 60, "sensor_status": "online"},

    # ---- Ativos adicionais (expansão para ~25, variedade de tipos/criticidade) ----
    # Forja Brasil (complemento)
    {"id": "asset_H110", "name": "Martelete hidráulico", "company_id": "comp_forja_br", "criticality": "high", "plant": "Planta 1", "line": "Forjamento", "parent_asset_id": None, "machine_type": "mill", "rotation_rpm": 220, "bearing_pn": "22224", "bpfo_hz": 38.2, "bpfi_hz": 55.1, "bsf_hz": 22.7, "ftf_hz": 4.5, "line_frequency_hz": None, "sensor_status": "online"},
    {"id": "asset_F115", "name": "Ventilador de resfriamento", "company_id": "comp_forja_br", "criticality": "medium", "plant": "Planta 1", "line": "Resfriamento", "parent_asset_id": None, "machine_type": "fan", "rotation_rpm": 980, "bearing_pn": "6312", "bpfo_hz": 69.4, "bpfi_hz": 100.8, "bsf_hz": 41.6, "ftf_hz": 8.1, "line_frequency_hz": None, "sensor_status": "online"},
    # Cervejaria Aurora (complemento)
    {"id": "asset_C210", "name": "Compressor de ar", "company_id": "comp_aurora", "criticality": "high", "plant": "Planta 1", "line": "Utilidades", "parent_asset_id": None, "machine_type": "compressor", "rotation_rpm": 2950, "bearing_pn": "7320", "bpfo_hz": 192.7, "bpfi_hz": 279.4, "bsf_hz": 99.3, "ftf_hz": 15.9, "line_frequency_hz": None, "sensor_status": "online"},
    {"id": "asset_B211", "name": "Bomba de refrigerante", "company_id": "comp_aurora", "criticality": "medium", "plant": "Planta 1", "line": "Envase", "parent_asset_id": None, "machine_type": "pump", "rotation_rpm": 1750, "bearing_pn": "6207", "bpfo_hz": 89.6, "bpfi_hz": 141.2, "bsf_hz": 59.4, "ftf_hz": 5.9, "line_frequency_hz": None, "sensor_status": "degraded"},
    # Papel Sul (complemento)
    {"id": "asset_R310", "name": "Rolo de prensa", "company_id": "comp_papel_sul", "criticality": "critical", "plant": "Planta 2", "line": "Prensagem", "parent_asset_id": None, "machine_type": "mill", "rotation_rpm": 180, "bearing_pn": "22228", "bpfo_hz": 31.4, "bpfi_hz": 45.3, "bsf_hz": 18.6, "ftf_hz": 3.7, "line_frequency_hz": None, "sensor_status": "online"},
    {"id": "asset_M312", "name": "Motor do refiner", "company_id": "comp_papel_sul", "criticality": "high", "plant": "Planta 2", "line": "Refino", "parent_asset_id": None, "machine_type": "motor_induction", "rotation_rpm": 1480, "bearing_pn": "6314", "bpfo_hz": 94.8, "bpfi_hz": 137.6, "bsf_hz": 57.7, "ftf_hz": 11.2, "line_frequency_hz": 60, "sensor_status": "online"},
    # Mineração Andes (complemento)
    {"id": "asset_C510", "name": "Britador cônico", "company_id": "comp_mineracao_andes", "criticality": "critical", "plant": "Mina Norte", "line": "Britagem", "parent_asset_id": None, "machine_type": "mill", "rotation_rpm": 280, "bearing_pn": "23230", "bpfo_hz": 48.6, "bpfi_hz": 70.2, "bsf_hz": 28.9, "ftf_hz": 5.8, "line_frequency_hz": None, "sensor_status": "online"},
    {"id": "asset_F520", "name": "Ventilador de exaustão", "company_id": "comp_mineracao_andes", "criticality": "medium", "plant": "Mina Norte", "line": "Ventilação", "parent_asset_id": None, "machine_type": "fan", "rotation_rpm": 1180, "bearing_pn": "22220", "bpfo_hz": 83.6, "bpfi_hz": 121.4, "bsf_hz": 49.8, "ftf_hz": 9.7, "line_frequency_hz": None, "sensor_status": "online"},
    # Petro Delta (complemento)
    {"id": "asset_P712", "name": "Bomba de transferência", "company_id": "comp_petro_delta", "criticality": "critical", "plant": "Plataforma A", "line": "Transferência", "parent_asset_id": None, "machine_type": "pump", "rotation_rpm": 3550, "bearing_pn": "7309", "bpfo_hz": 168.3, "bpfi_hz": 244.1, "bsf_hz": 86.9, "ftf_hz": 13.9, "line_frequency_hz": None, "sensor_status": "online"},
    {"id": "asset_G715", "name": "Gerador", "company_id": "comp_petro_delta", "criticality": "critical", "plant": "Plataforma A", "line": "Geração", "parent_asset_id": None, "machine_type": "motor_induction", "rotation_rpm": 1800, "bearing_pn": "6322", "bpfo_hz": 117.2, "bpfi_hz": 170.1, "bsf_hz": 71.3, "ftf_hz": 14.1, "line_frequency_hz": 60, "sensor_status": "online"},
    # Acme Auto Peças (complemento)
    {"id": "asset_S425", "name": "Spindle secundário", "company_id": "comp_acme", "criticality": "medium", "plant": "Planta 1", "line": "Usinagem", "parent_asset_id": None, "machine_type": "spindle", "rotation_rpm": 9000, "bearing_pn": "7006", "bpfo_hz": 541.2, "bpfi_hz": 823.7, "bsf_hz": 234.6, "ftf_hz": 36.1, "line_frequency_hz": None, "sensor_status": "online"},
    {"id": "asset_M428", "name": "Motor de mesa", "company_id": "comp_acme", "criticality": "low", "plant": "Planta 1", "line": "Usinagem", "parent_asset_id": None, "machine_type": "motor_induction", "rotation_rpm": 1480, "bearing_pn": "6208", "bpfo_hz": 76.8, "bpfi_hz": 121.1, "bsf_hz": 50.9, "ftf_hz": 10.1, "line_frequency_hz": 60, "sensor_status": "online"},
    # Cimento Vale (complemento)
    {"id": "asset_F215", "name": "Ventilador do forno", "company_id": "comp_cimento_vale", "criticality": "critical", "plant": "Planta 1", "line": "Queima", "parent_asset_id": None, "machine_type": "fan", "rotation_rpm": 1180, "bearing_pn": "22224", "bpfo_hz": 78.4, "bpfi_hz": 113.9, "bsf_hz": 46.7, "ftf_hz": 9.1, "line_frequency_hz": None, "sensor_status": "online"},
    {"id": "asset_X216", "name": "Misturador de cru", "company_id": "comp_cimento_vale", "criticality": "medium", "plant": "Planta 1", "line": "Moagem", "parent_asset_id": None, "machine_type": "mill", "rotation_rpm": 320, "bearing_pn": "22226", "bpfo_hz": 55.3, "bpfi_hz": 80.1, "bsf_hz": 32.9, "ftf_hz": 6.6, "line_frequency_hz": None, "sensor_status": "online"},
    # Texfil (complemento)
    {"id": "asset_R610", "name": "Roteador de tecido", "company_id": "comp_texfil", "criticality": "medium", "plant": "Planta 1", "line": "Tecelagem", "parent_asset_id": None, "machine_type": "mill", "rotation_rpm": 260, "bearing_pn": "6209", "bpfo_hz": 47.2, "bpfi_hz": 74.5, "bsf_hz": 31.3, "ftf_hz": 6.2, "line_frequency_hz": None, "sensor_status": "online"},
    {"id": "asset_M612", "name": "Motor de bobina", "company_id": "comp_texfil", "criticality": "high", "plant": "Planta 1", "line": "Bobinagem", "parent_asset_id": None, "machine_type": "motor_induction", "rotation_rpm": 1480, "bearing_pn": "6309", "bpfo_hz": 89.1, "bpfi_hz": 136.4, "bsf_hz": 57.2, "ftf_hz": 11.3, "line_frequency_hz": 60, "sensor_status": "online"},
]

POINTS = [
    {"id": "pt_M101_de", "asset_id": "asset_M101", "location": "DE", "sensor_status": "online"},
    {"id": "pt_M101_nde", "asset_id": "asset_M101", "location": "NDE", "sensor_status": "online"},
    {"id": "pt_M102_de", "asset_id": "asset_M102", "location": "DE", "sensor_status": "online"},
    {"id": "pt_B204_de", "asset_id": "asset_B204", "location": "DE", "sensor_status": "online"},
    {"id": "pt_V301_de", "asset_id": "asset_V301", "location": "DE", "sensor_status": "degraded"},
    {"id": "pt_G501_de", "asset_id": "asset_G501", "location": "DE", "sensor_status": "offline"},
    {"id": "pt_C710_de", "asset_id": "asset_C710", "location": "DE", "sensor_status": "online"},
    {"id": "pt_S420_de", "asset_id": "asset_S420", "location": "DE", "sensor_status": "online"},
    {"id": "pt_M205_de", "asset_id": "asset_M205", "location": "DE", "sensor_status": "online"},
    {"id": "pt_M208_de", "asset_id": "asset_M208", "location": "DE", "sensor_status": "online"},
    {"id": "pt_M605_de", "asset_id": "asset_M605", "location": "DE", "sensor_status": "online"},
    # Points dos ativos adicionais
    {"id": "pt_H110_de", "asset_id": "asset_H110", "location": "DE", "sensor_status": "online"},
    {"id": "pt_F115_de", "asset_id": "asset_F115", "location": "DE", "sensor_status": "online"},
    {"id": "pt_C210_de", "asset_id": "asset_C210", "location": "DE", "sensor_status": "online"},
    {"id": "pt_B211_de", "asset_id": "asset_B211", "location": "DE", "sensor_status": "degraded"},
    {"id": "pt_R310_de", "asset_id": "asset_R310", "location": "DE", "sensor_status": "online"},
    {"id": "pt_M312_de", "asset_id": "asset_M312", "location": "DE", "sensor_status": "online"},
    {"id": "pt_C510_de", "asset_id": "asset_C510", "location": "DE", "sensor_status": "online"},
    {"id": "pt_F520_de", "asset_id": "asset_F520", "location": "DE", "sensor_status": "online"},
    {"id": "pt_P712_de", "asset_id": "asset_P712", "location": "DE", "sensor_status": "online"},
    {"id": "pt_G715_de", "asset_id": "asset_G715", "location": "DE", "sensor_status": "online"},
    {"id": "pt_S425_de", "asset_id": "asset_S425", "location": "DE", "sensor_status": "online"},
    {"id": "pt_M428_de", "asset_id": "asset_M428", "location": "DE", "sensor_status": "online"},
    {"id": "pt_F215_de", "asset_id": "asset_F215", "location": "DE", "sensor_status": "online"},
    {"id": "pt_X216_de", "asset_id": "asset_X216", "location": "DE", "sensor_status": "online"},
    {"id": "pt_R610_de", "asset_id": "asset_R610", "location": "DE", "sensor_status": "online"},
    {"id": "pt_M612_de", "asset_id": "asset_M612", "location": "DE", "sensor_status": "online"},
]


# ---------------------------------------------------------------------------
# Baselines (estado normal aprendido). detection_mode: baseline | symptom
# ---------------------------------------------------------------------------
BASELINES = [
    {"id": "bs_M101_de", "asset_id": "asset_M101", "point_id": "pt_M101_de", "state": "established", "detection_mode": "baseline", "learnable": True, "established_at": ts(days=-60), "invalidated_at": None, "invalidation_reason": None, "features": '[{"feature":"rms_mm_s","reference":2.1,"tolerance":0.9},{"feature":"bpfo_amplitude","reference":0.4,"tolerance":0.3}]'},
    {"id": "bs_M102_de", "asset_id": "asset_M102", "point_id": "pt_M102_de", "state": "learning", "detection_mode": "baseline", "learnable": False, "established_at": None, "invalidated_at": None, "invalidation_reason": None, "features": "[]"},
    {"id": "bs_B204_de", "asset_id": "asset_B204", "point_id": "pt_B204_de", "state": "invalidated", "detection_mode": "baseline", "learnable": True, "established_at": ts(days=-120), "invalidated_at": ts(days=-4), "invalidation_reason": "maintenance_intervention", "features": '[{"feature":"rms_mm_s","reference":2.4,"tolerance":1.0},{"feature":"bpfo_amplitude","reference":0.5,"tolerance":0.3}]'},
    {"id": "bs_V301_de", "asset_id": "asset_V301", "point_id": "pt_V301_de", "state": "established", "detection_mode": "baseline", "learnable": True, "established_at": ts(days=-90), "invalidated_at": None, "invalidation_reason": None, "features": '[{"feature":"rms_mm_s","reference":3.2,"tolerance":1.4}]'},
    {"id": "bs_G501_de", "asset_id": "asset_G501", "point_id": "pt_G501_de", "state": "learning", "detection_mode": "baseline", "learnable": False, "established_at": None, "invalidated_at": None, "invalidation_reason": None, "features": "[]"},
    {"id": "bs_C710_de", "asset_id": "asset_C710", "point_id": "pt_C710_de", "state": "established", "detection_mode": "baseline", "learnable": True, "established_at": ts(days=-180), "invalidated_at": None, "invalidation_reason": None, "features": '[{"feature":"rms_mm_s","reference":1.8,"tolerance":0.8},{"feature":"bpfo_amplitude","reference":0.6,"tolerance":0.4}]'},
    {"id": "bs_S420_de", "asset_id": "asset_S420", "point_id": "pt_S420_de", "state": "invalidated", "detection_mode": "baseline", "learnable": True, "established_at": ts(days=-200), "invalidated_at": ts(days=-8), "invalidation_reason": "maintenance_intervention", "features": '[{"feature":"rms_mm_s","reference":1.2,"tolerance":0.6}]'},
    {"id": "bs_M205_de", "asset_id": "asset_M205", "point_id": "pt_M205_de", "state": "established", "detection_mode": "baseline", "learnable": True, "established_at": ts(days=-150), "invalidated_at": None, "invalidation_reason": None, "features": '[{"feature":"rms_mm_s","reference":2.8,"tolerance":1.2}]'},
    {"id": "bs_M208_de", "asset_id": "asset_M208", "point_id": "pt_M208_de", "state": "learning", "detection_mode": "symptom", "learnable": False, "established_at": None, "invalidated_at": None, "invalidation_reason": None, "features": "[]"},
    {"id": "bs_M605_de", "asset_id": "asset_M605", "point_id": "pt_M605_de", "state": "established", "detection_mode": "baseline", "learnable": True, "established_at": ts(days=-100), "invalidated_at": None, "invalidation_reason": None, "features": '[{"feature":"rms_mm_s","reference":1.9,"tolerance":0.8}]'},
    # Baselines dos ativos adicionais (majority established; alguns learning/invalidated p/ variedade)
    {"id": "bs_H110_de", "asset_id": "asset_H110", "point_id": "pt_H110_de", "state": "established", "detection_mode": "baseline", "learnable": True, "established_at": ts(days=-70), "invalidated_at": None, "invalidation_reason": None, "features": '[{"feature":"rms_mm_s","reference":2.6,"tolerance":1.1}]'},
    {"id": "bs_F115_de", "asset_id": "asset_F115", "point_id": "pt_F115_de", "state": "established", "detection_mode": "baseline", "learnable": True, "established_at": ts(days=-80), "invalidated_at": None, "invalidation_reason": None, "features": '[{"feature":"rms_mm_s","reference":3.0,"tolerance":1.3}]'},
    {"id": "bs_C210_de", "asset_id": "asset_C210", "point_id": "pt_C210_de", "state": "established", "detection_mode": "baseline", "learnable": True, "established_at": ts(days=-110), "invalidated_at": None, "invalidation_reason": None, "features": '[{"feature":"rms_mm_s","reference":1.7,"tolerance":0.7}]'},
    {"id": "bs_B211_de", "asset_id": "asset_B211", "point_id": "pt_B211_de", "state": "invalidated", "detection_mode": "baseline", "learnable": True, "established_at": ts(days=-60), "invalidated_at": ts(days=-2), "invalidation_reason": "config_change", "features": '[{"feature":"rms_mm_s","reference":2.2,"tolerance":0.9}]'},
    {"id": "bs_R310_de", "asset_id": "asset_R310", "point_id": "pt_R310_de", "state": "established", "detection_mode": "baseline", "learnable": True, "established_at": ts(days=-130), "invalidated_at": None, "invalidation_reason": None, "features": '[{"feature":"rms_mm_s","reference":2.9,"tolerance":1.2}]'},
    {"id": "bs_M312_de", "asset_id": "asset_M312", "point_id": "pt_M312_de", "state": "established", "detection_mode": "baseline", "learnable": True, "established_at": ts(days=-95), "invalidated_at": None, "invalidation_reason": None, "features": '[{"feature":"rms_mm_s","reference":2.3,"tolerance":1.0}]'},
    {"id": "bs_C510_de", "asset_id": "asset_C510", "point_id": "pt_C510_de", "state": "learning", "detection_mode": "baseline", "learnable": False, "established_at": None, "invalidated_at": None, "invalidation_reason": None, "features": "[]"},
    {"id": "bs_F520_de", "asset_id": "asset_F520", "point_id": "pt_F520_de", "state": "established", "detection_mode": "baseline", "learnable": True, "established_at": ts(days=-85), "invalidated_at": None, "invalidation_reason": None, "features": '[{"feature":"rms_mm_s","reference":3.1,"tolerance":1.3}]'},
    {"id": "bs_P712_de", "asset_id": "asset_P712", "point_id": "pt_P712_de", "state": "established", "detection_mode": "baseline", "learnable": True, "established_at": ts(days=-140), "invalidated_at": None, "invalidation_reason": None, "features": '[{"feature":"rms_mm_s","reference":1.6,"tolerance":0.7}]'},
    {"id": "bs_G715_de", "asset_id": "asset_G715", "point_id": "pt_G715_de", "state": "established", "detection_mode": "baseline", "learnable": True, "established_at": ts(days=-160), "invalidated_at": None, "invalidation_reason": None, "features": '[{"feature":"rms_mm_s","reference":2.0,"tolerance":0.9}]'},
    {"id": "bs_S425_de", "asset_id": "asset_S425", "point_id": "pt_S425_de", "state": "established", "detection_mode": "baseline", "learnable": True, "established_at": ts(days=-40), "invalidated_at": None, "invalidation_reason": None, "features": '[{"feature":"rms_mm_s","reference":1.1,"tolerance":0.5}]'},
    {"id": "bs_M428_de", "asset_id": "asset_M428", "point_id": "pt_M428_de", "state": "learning", "detection_mode": "baseline", "learnable": True, "established_at": None, "invalidated_at": None, "invalidation_reason": None, "features": "[]"},
    {"id": "bs_F215_de", "asset_id": "asset_F215", "point_id": "pt_F215_de", "state": "established", "detection_mode": "baseline", "learnable": True, "established_at": ts(days=-150), "invalidated_at": None, "invalidation_reason": None, "features": '[{"feature":"rms_mm_s","reference":3.4,"tolerance":1.5}]'},
    {"id": "bs_X216_de", "asset_id": "asset_X216", "point_id": "pt_X216_de", "state": "established", "detection_mode": "baseline", "learnable": True, "established_at": ts(days=-65), "invalidated_at": None, "invalidation_reason": None, "features": '[{"feature":"rms_mm_s","reference":2.7,"tolerance":1.1}]'},
    {"id": "bs_R610_de", "asset_id": "asset_R610", "point_id": "pt_R610_de", "state": "established", "detection_mode": "baseline", "learnable": True, "established_at": ts(days=-75), "invalidated_at": None, "invalidation_reason": None, "features": '[{"feature":"rms_mm_s","reference":2.4,"tolerance":1.0}]'},
    {"id": "bs_M612_de", "asset_id": "asset_M612", "point_id": "pt_M612_de", "state": "established", "detection_mode": "baseline", "learnable": True, "established_at": ts(days=-90), "invalidated_at": None, "invalidation_reason": None, "features": '[{"feature":"rms_mm_s","reference":2.2,"tolerance":0.9}]'},
]


# ---------------------------------------------------------------------------
# Análises (insights)
# ---------------------------------------------------------------------------
ANALYSES = [
    # G-501: inconclusive, sem baseline
    {"id": "an_9901", "asset_id": "asset_G501", "point_id": "pt_G501_de", "type": "bearing_fault", "detection_mode": "baseline", "severity": "none", "confidence": 0.2, "baseline_state_at_detection": "learning", "evidence": "[]", "limitations": '["baseline_learning","data_gap"]', "model_version": "3.2.1", "created_at": ts(days=-1), "status": "inconclusive"},
    # C-710: pending, baseline established, RMS subindo
    {"id": "an_9902", "asset_id": "asset_C710", "point_id": "pt_C710_de", "type": "bearing_fault", "detection_mode": "baseline", "severity": "high", "confidence": 0.78, "baseline_state_at_detection": "established", "evidence": '[{"metric":"bpfo_amplitude","value":1.4,"reference":0.6,"note":"BPFO 2.3x o baseline"}]', "limitations": '["processing_delayed"]', "model_version": "3.2.1", "created_at": ts(days=-3), "status": "pending"},
    # S-420: falso positivo, baseline invalidated
    {"id": "an_9903", "asset_id": "asset_S420", "point_id": "pt_S420_de", "type": "imbalance", "detection_mode": "baseline", "severity": "medium", "confidence": 0.81, "baseline_state_at_detection": "invalidated", "evidence": '[{"metric":"1x_amplitude","value":1.6,"reference":0.9,"note":"1x acima do baseline velho"}]', "limitations": '["baseline_invalidated"]', "model_version": "3.2.1", "created_at": ts(days=-2), "status": "current"},
    # S-420: análise especializada (conflito) — looseness
    {"id": "an_9904", "asset_id": "asset_S420", "point_id": "pt_S420_de", "type": "looseness", "detection_mode": "baseline", "severity": "low", "confidence": 0.66, "baseline_state_at_detection": "established", "evidence": '[{"metric":"subharmonics","value":0.7,"reference":0.2,"note":"subharmônicos presentes"}]', "limitations": "[]", "model_version": "specialist_v1", "created_at": ts(days=-5), "status": "current"},
    # M-208: lubrificação sintomática, baseline learning
    {"id": "an_9905", "asset_id": "asset_M208", "point_id": "pt_M208_de", "type": "lubrication", "detection_mode": "symptom", "severity": "medium", "confidence": 0.72, "baseline_state_at_detection": "not_applicable", "evidence": '[{"metric":"shock_pulse","value":2.8,"reference":null,"note":"pulsos de choque/atrito típicos de falta de lubrificação"}]', "limitations": "[]", "model_version": "3.2.1", "created_at": ts(days=-1), "status": "current"},
    # B-204: stale pós-manutenção
    {"id": "an_9906", "asset_id": "asset_B204", "point_id": "pt_B204_de", "type": "bearing_fault", "detection_mode": "baseline", "severity": "high", "confidence": 0.75, "baseline_state_at_detection": "invalidated", "evidence": '[{"metric":"bpfo_amplitude","value":1.1,"reference":0.5,"note":"BPFO alto vs baseline pré-manutenção"}]', "limitations": '["baseline_invalidated"]', "model_version": "3.2.1", "created_at": ts(days=-6), "status": "stale"},
    # M-205: conflito misalignment vs looseness
    {"id": "an_9907", "asset_id": "asset_M205", "point_id": "pt_M205_de", "type": "misalignment", "detection_mode": "baseline", "severity": "medium", "confidence": 0.69, "baseline_state_at_detection": "established", "evidence": '[{"metric":"2x_amplitude","value":1.3,"reference":0.5,"note":"2x dominante"}]', "limitations": "[]", "model_version": "3.2.1", "created_at": ts(days=-2), "status": "current"},
    {"id": "an_9908", "asset_id": "asset_M205", "point_id": "pt_M205_de", "type": "looseness", "detection_mode": "baseline", "severity": "medium", "confidence": 0.71, "baseline_state_at_detection": "established", "evidence": '[{"metric":"subharmonics","value":0.9,"reference":0.2,"note":"subharmônicos e 0.5x"}]', "limitations": "[]", "model_version": "specialist_v1", "created_at": ts(days=-3), "status": "current"},
    # V-301: confiança alta com qualidade baixa
    {"id": "an_9909", "asset_id": "asset_V301", "point_id": "pt_V301_de", "type": "imbalance", "detection_mode": "baseline", "severity": "medium", "confidence": 0.83, "baseline_state_at_detection": "established", "evidence": '[{"metric":"1x_amplitude","value":2.1,"reference":1.0,"note":"1x acima do baseline"}]', "limitations": '["low_signal_quality"]', "model_version": "3.2.1", "created_at": ts(days=-1), "status": "current"},
    # M-605: vibração abrupta, banda de 2x f-linha ausente -> não confirma elétrica
    {"id": "an_9910", "asset_id": "asset_M605", "point_id": "pt_M605_de", "type": "electrical_fault", "detection_mode": "baseline", "severity": "low", "confidence": 0.41, "baseline_state_at_detection": "established", "evidence": '[{"metric":"rms_mm_s","value":2.7,"reference":1.9,"note":"RMS acima do baseline; banda de 2x f-linha não disponível para confirmar elétrica"}]', "limitations": '["partial_spectrum","band_2x_line_missing"]', "model_version": "3.2.1", "created_at": ts(days=-1), "status": "inconclusive"},
]


# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------
MODELS = [
    {"id": "mdl_vib_v3", "version": "3.2.1", "coverage": '[{"machine_type":"motor_induction","supported":true,"can_learn_baseline":true},{"machine_type":"motor_dc","supported":true,"can_learn_baseline":false,"note":"detecção apenas sintomática"},{"machine_type":"pump","supported":true,"can_learn_baseline":true},{"machine_type":"compressor","supported":true,"can_learn_baseline":true},{"machine_type":"fan","supported":true,"can_learn_baseline":true},{"machine_type":"spindle","supported":true,"can_learn_baseline":true},{"machine_type":"gearbox","supported":true,"can_learn_baseline":false,"note":"baixa rotação: só sintomática"},{"machine_type":"mill","supported":true,"can_learn_baseline":true}]', "min_completeness": 0.8, "min_snr_db": 12.0, "min_rotation_rpm": None, "processing_state": "delayed", "last_run_at": ts(days=-2)},
]


# ---------------------------------------------------------------------------
# Conhecimento
# ---------------------------------------------------------------------------
KNOWLEDGE = [
    {"id": "kb_proc_001", "type": "procedure", "title": "Troca de rolamento em motor industrial", "body": "## Procedimento\n1. Isolar eletricamente o motor.\n2. Remover o rolamento antigo com extrator hidráulico.\n3. Aquecer o rolamento novo (indução, máx 110°C) e montar.\n4. Torque de aperto conforme catálogo do fabricante.\n5. Reaprender o baseline de vibração após 24h de operação sadía.\n\n> Após a troca, o baseline anterior é **invalidado** (maintenance_intervention) e o modelo entra em `learning` até estabelecer nova referência.", "tags": '["rolamento","manutencao","baseline"]'},
    {"id": "kb_glos_001", "type": "glossary", "title": "BPFO (Ball Pass Frequency Outer)", "body": "**BPFO** é a frequência característica de defeito na pista externa de um rolamento. Quando um rolamento apresenta dano na pista externa, pulsos de vibração surgem a essa frequência (e seus múltiplos) no espectro. Uma amplitude crescente em BPFO acima do baseline indica falha externa de rolamento.", "tags": '["rolamento","fft","glossario"]'},
    {"id": "kb_guid_001", "type": "guidance", "title": "Interpretando limiares de RMS", "body": "O limiar de alarme de RMS **não** é uma tabela fixa por classe de máquina. Ele é derivado do **baseline** aprendido do próprio ativo: `alarme = reference + tolerance`. Assim, o que é alarme para um ventilador sadio (RMS ~3.2) não é o mesmo para um motor leve (RMS ~2.1). Quando o baseline está em `learning`, não há limiar confiável.", "tags": '["rms","baseline","alarme"]'},
    {"id": "kb_guid_002", "type": "guidance", "title": "Detecção sintomática vs. por desvio", "body": "Falhas por **desvio** (imbalance, misalignment, bearing_fault, electrical_fault) exigem um baseline `established`: comparam o sinal atual ao estado normal aprendido. Falhas **sintomáticas** (lubrificação) não precisam de baseline: a presença do sintoma (pulsos de choque/atrito) já indica a falha — podem ser detectadas mesmo com baseline em `learning`.", "tags": '["baseline","sintomatica","lubrificacao"]'},
    {"id": "kb_guid_003", "type": "guidance", "title": "Falhas elétricas em motores", "body": "No espectro de vibração, falhas elétricas (ex.: barras quebradas, excentricidade) costumam gerar componentes em **2x a frequência de linha** (120 Hz em 60 Hz). Diferenciar de falhas mecânicas exige inspecionar a banda ao redor de 2x f-linha. Espectros parciais nessa banda tornam a inferência incerta.", "tags": '["eletrica","fft","motor"]'},
]


# ---------------------------------------------------------------------------
# Séries RMS (geradas proceduralmente, coerentes com as análises)
# ---------------------------------------------------------------------------
def _gen_rms(asset_id: str, point_id: str, baseline_ref: float | None, trend: str,
             n_days: int = 30, noise: float = 0.15) -> list[dict]:
    """Gera série RMS diária. trend: flat | rising | drop | gap | noisy."""
    rows = []
    for i in range(n_days):
        day = -(n_days - 1 - i)  # do mais antigo ao mais recente
        if trend == "gap" and i >= n_days - 5:
            # sensor mudo nos últimos 5 dias (G-501 antes da quebra)
            continue
        if trend == "noisy":
            val = (baseline_ref or 2.0) + RNG.gauss(0, noise * 3)
        elif trend == "rising":
            # subida gradual + aceleração no fim (C-710)
            frac = i / (n_days - 1)
            val = (baseline_ref or 1.8) * (1 + 0.6 * frac + 0.2 * frac ** 2) + RNG.gauss(0, noise)
        elif trend == "drop":
            # queda pós-manutenção (B-204): alto até dia -4, depois cai
            if day < -4:
                val = (baseline_ref or 2.4) * 1.6 + RNG.gauss(0, noise)
            else:
                val = (baseline_ref or 2.4) * 0.8 + RNG.gauss(0, noise)
        else:  # flat
            val = (baseline_ref or 2.0) + RNG.gauss(0, noise)
        rows.append({"asset_id": asset_id, "point_id": point_id, "ts": ts(days=day), "value": round(max(0.1, val), 3)})
    return rows


RMS_ROWS = (
    _gen_rms("asset_M101", "pt_M101_de", 2.1, "flat")
    + _gen_rms("asset_M102", "pt_M102_de", None, "noisy")
    + _gen_rms("asset_B204", "pt_B204_de", 2.4, "drop")
    + _gen_rms("asset_V301", "pt_V301_de", 3.2, "noisy")
    + _gen_rms("asset_G501", "pt_G501_de", None, "gap")
    + _gen_rms("asset_C710", "pt_C710_de", 1.8, "rising")
    + _gen_rms("asset_S420", "pt_S420_de", 1.2, "flat")
    + _gen_rms("asset_M205", "pt_M205_de", 2.8, "flat")
    + _gen_rms("asset_M208", "pt_M208_de", None, "noisy")
    + _gen_rms("asset_M605", "pt_M605_de", 1.9, "flat")
    + _gen_rms("asset_H110", "pt_H110_de", 2.6, "flat")
    + _gen_rms("asset_F115", "pt_F115_de", 3.0, "flat")
    + _gen_rms("asset_C210", "pt_C210_de", 1.7, "flat")
    + _gen_rms("asset_B211", "pt_B211_de", 2.2, "noisy")
    + _gen_rms("asset_R310", "pt_R310_de", 2.9, "flat")
    + _gen_rms("asset_M312", "pt_M312_de", 2.3, "flat")
    + _gen_rms("asset_C510", "pt_C510_de", None, "noisy")
    + _gen_rms("asset_F520", "pt_F520_de", 3.1, "flat")
    + _gen_rms("asset_P712", "pt_P712_de", 1.6, "flat")
    + _gen_rms("asset_G715", "pt_G715_de", 2.0, "flat")
    + _gen_rms("asset_S425", "pt_S425_de", 1.1, "flat")
    + _gen_rms("asset_M428", "pt_M428_de", None, "noisy")
    + _gen_rms("asset_F215", "pt_F215_de", 3.4, "flat")
    + _gen_rms("asset_X216", "pt_X216_de", 2.7, "flat")
    + _gen_rms("asset_R610", "pt_R610_de", 2.4, "flat")
    + _gen_rms("asset_M612", "pt_M612_de", 2.2, "flat")
)


# ---------------------------------------------------------------------------
# Espectros (picos relevantes por ativo)
# ---------------------------------------------------------------------------
SPECTRA = [
    {"asset_id": "asset_M101", "point_id": "pt_M101_de", "collected_at": ts(days=-1), "peaks": '[{"freq_hz":29.7,"amplitude_mm_s":0.8,"note":"1x"},{"freq_hz":59.4,"amplitude_mm_s":0.3,"note":"2x"},{"freq_hz":142.3,"amplitude_mm_s":0.4,"note":"BPFO"}]', "bands_missing": "[]"},
    {"asset_id": "asset_B204", "point_id": "pt_B204_de", "collected_at": ts(days=-1), "peaks": '[{"freq_hz":29.2,"amplitude_mm_s":0.7,"note":"1x"},{"freq_hz":107.4,"amplitude_mm_s":0.5,"note":"BPFO"}]', "bands_missing": "[]"},
    {"asset_id": "asset_V301", "point_id": "pt_V301_de", "collected_at": ts(days=-1), "peaks": '[{"freq_hz":19.7,"amplitude_mm_s":2.1,"note":"1x"},{"freq_hz":83.6,"amplitude_mm_s":0.4,"note":"BPFO"}]', "bands_missing": "[]"},
    {"asset_id": "asset_G501", "point_id": "pt_G501_de", "collected_at": ts(days=-2), "peaks": "[]", "bands_missing": '["full_spectrum"]'},
    {"asset_id": "asset_C710", "point_id": "pt_C710_de", "collected_at": ts(days=-1), "peaks": '[{"freq_hz":49.2,"amplitude_mm_s":0.6,"note":"1x"},{"freq_hz":198.2,"amplitude_mm_s":1.4,"note":"BPFO"}]', "bands_missing": "[]"},
    {"asset_id": "asset_S420", "point_id": "pt_S420_de", "collected_at": ts(days=-1), "peaks": '[{"freq_hz":200,"amplitude_mm_s":1.6,"note":"1x"},{"freq_hz":400,"amplitude_mm_s":0.4,"note":"2x"},{"freq_hz":300,"amplitude_mm_s":0.7,"note":"subharmônico (looseness)"}]', "bands_missing": "[]"},
    {"asset_id": "asset_M205", "point_id": "pt_M205_de", "collected_at": ts(days=-1), "peaks": '[{"freq_hz":4.0,"amplitude_mm_s":1.3,"note":"2x"},{"freq_hz":2.0,"amplitude_mm_s":0.9,"note":"0.5x/subharmônico (looseness)"}]', "bands_missing": "[]"},
    {"asset_id": "asset_M208", "point_id": "pt_M208_de", "collected_at": ts(days=-1), "peaks": '[{"freq_hz":4500,"amplitude_mm_s":2.8,"note":"shock pulse (atrito/lubrificação)"}]', "bands_missing": "[]"},
    {"asset_id": "asset_M605", "point_id": "pt_M605_de", "collected_at": ts(days=-1), "peaks": '[{"freq_hz":59.2,"amplitude_mm_s":0.5,"note":"1x"}]', "bands_missing": '["120-140hz (2x f-linha)"]'},
]


# ---------------------------------------------------------------------------
# Qualidade dos dados
# ---------------------------------------------------------------------------
DATA_QUALITY = [
    {"asset_id": "asset_M101", "point_id": "pt_M101_de", "completeness": 0.98, "freshness_minutes": 5, "snr_db": 18.2, "staleness_flag": False},
    {"asset_id": "asset_M102", "point_id": "pt_M102_de", "completeness": 0.90, "freshness_minutes": 8, "snr_db": 14.1, "staleness_flag": False},
    {"asset_id": "asset_B204", "point_id": "pt_B204_de", "completeness": 0.95, "freshness_minutes": 6, "snr_db": 16.5, "staleness_flag": False},
    {"asset_id": "asset_V301", "point_id": "pt_V301_de", "completeness": 0.62, "freshness_minutes": 240, "snr_db": 8.4, "staleness_flag": True},
    {"asset_id": "asset_G501", "point_id": "pt_G501_de", "completeness": 0.18, "freshness_minutes": 7200, "snr_db": 3.1, "staleness_flag": True},
    {"asset_id": "asset_C710", "point_id": "pt_C710_de", "completeness": 0.93, "freshness_minutes": 7, "snr_db": 17.0, "staleness_flag": False},
    {"asset_id": "asset_S420", "point_id": "pt_S420_de", "completeness": 0.97, "freshness_minutes": 4, "snr_db": 19.1, "staleness_flag": False},
    {"asset_id": "asset_M205", "point_id": "pt_M205_de", "completeness": 0.91, "freshness_minutes": 9, "snr_db": 15.6, "staleness_flag": False},
    {"asset_id": "asset_M208", "point_id": "pt_M208_de", "completeness": 0.88, "freshness_minutes": 10, "snr_db": 13.8, "staleness_flag": False},
    {"asset_id": "asset_M605", "point_id": "pt_M605_de", "completeness": 0.70, "freshness_minutes": 30, "snr_db": 11.2, "staleness_flag": False},
    # Ativos adicionais (majority sadia; B211/C510 degradados p/ variedade)
    {"asset_id": "asset_H110", "point_id": "pt_H110_de", "completeness": 0.95, "freshness_minutes": 6, "snr_db": 16.1, "staleness_flag": False},
    {"asset_id": "asset_F115", "point_id": "pt_F115_de", "completeness": 0.94, "freshness_minutes": 7, "snr_db": 15.8, "staleness_flag": False},
    {"asset_id": "asset_C210", "point_id": "pt_C210_de", "completeness": 0.96, "freshness_minutes": 5, "snr_db": 17.2, "staleness_flag": False},
    {"asset_id": "asset_B211", "point_id": "pt_B211_de", "completeness": 0.58, "freshness_minutes": 180, "snr_db": 7.9, "staleness_flag": True},
    {"asset_id": "asset_R310", "point_id": "pt_R310_de", "completeness": 0.92, "freshness_minutes": 8, "snr_db": 15.2, "staleness_flag": False},
    {"asset_id": "asset_M312", "point_id": "pt_M312_de", "completeness": 0.97, "freshness_minutes": 4, "snr_db": 18.0, "staleness_flag": False},
    {"asset_id": "asset_C510", "point_id": "pt_C510_de", "completeness": 0.55, "freshness_minutes": 300, "snr_db": 8.1, "staleness_flag": True},
    {"asset_id": "asset_F520", "point_id": "pt_F520_de", "completeness": 0.93, "freshness_minutes": 6, "snr_db": 16.0, "staleness_flag": False},
    {"asset_id": "asset_P712", "point_id": "pt_P712_de", "completeness": 0.96, "freshness_minutes": 5, "snr_db": 17.5, "staleness_flag": False},
    {"asset_id": "asset_G715", "point_id": "pt_G715_de", "completeness": 0.94, "freshness_minutes": 7, "snr_db": 16.4, "staleness_flag": False},
    {"asset_id": "asset_S425", "point_id": "pt_S425_de", "completeness": 0.95, "freshness_minutes": 6, "snr_db": 17.8, "staleness_flag": False},
    {"asset_id": "asset_M428", "point_id": "pt_M428_de", "completeness": 0.91, "freshness_minutes": 9, "snr_db": 14.8, "staleness_flag": False},
    {"asset_id": "asset_F215", "point_id": "pt_F215_de", "completeness": 0.92, "freshness_minutes": 8, "snr_db": 15.9, "staleness_flag": False},
    {"asset_id": "asset_X216", "point_id": "pt_X216_de", "completeness": 0.93, "freshness_minutes": 7, "snr_db": 15.5, "staleness_flag": False},
    {"asset_id": "asset_R610", "point_id": "pt_R610_de", "completeness": 0.90, "freshness_minutes": 10, "snr_db": 14.5, "staleness_flag": False},
    {"asset_id": "asset_M612", "point_id": "pt_M612_de", "completeness": 0.94, "freshness_minutes": 6, "snr_db": 16.2, "staleness_flag": False},
]


# ---------------------------------------------------------------------------
# Casos (unifica chamados)
# ---------------------------------------------------------------------------
CASES = [
    {"id": "case_tkt_inv_04", "ticket_id": "TKT-INV-04", "company_id": "comp_mineracao_andes", "user_id": "usr_pedro", "asset_id": "asset_G501", "message": "O redutor da correia transportadora quebrou ontem e eu não recebi nenhum aviso. Por quê?", "root_question": "Por que nenhum insight/notificação foi gerado antes da quebra?", "mode": "inconclusive", "expected_path": '[{"step":"GET /assets/asset_G501","note":"config, sensor_status=offline"},{"step":"GET /assets/asset_G501/baseline","note":"state=learning"},{"step":"GET /assets/asset_G501/data-quality","note":"completeness baixa, gap"},{"step":"GET /assets/asset_G501/rms","note":"mode=unavailable"},{"step":"POST /cases/case_tkt_inv_04/escalate","note":"escalar para humano"}]'},
    {"id": "case_tkt_inv_05", "ticket_id": "TKT-INV-05", "company_id": "comp_petro_delta", "user_id": "usr_sofia", "asset_id": "asset_C710", "message": "Tô vendo o RMS do compressor subindo há duas semanas, mas não recebi insight nenhum. Cadê o diagnóstico?", "root_question": "Por que não há insight apesar da tendência de RMS?", "mode": "pending", "expected_path": '[{"step":"GET /assets/asset_C710/rms","note":"tendência + alarm_threshold ultrapassado"},{"step":"GET /assets/asset_C710/baseline","note":"established"},{"step":"GET /assets/asset_C710/analyses?status=pending","note":"status=pending"},{"step":"GET /models/mdl_vib_v3","note":"processing_state=delayed"}]'},
    {"id": "case_tkt_inv_06", "ticket_id": "TKT-INV-06", "company_id": "comp_acme", "user_id": "usr_bruno", "asset_id": "asset_S420", "message": "Recebi um insight dizendo desbalanceamento no spindle, mas a máquina tá rodando lisa. Isso não é nada.", "root_question": "O insight é falso positivo?", "mode": "conflict", "expected_path": '[{"step":"GET /analyses/an_9903","note":"imbalance, baseline_state_at_detection=invalidated"},{"step":"GET /assets/asset_S420/baseline","note":"state=invalidated"},{"step":"GET /assets/asset_S420/spectrum","note":"1x baixo não sustenta desbalanceamento"},{"step":"GET /analyses/an_9904","note":"conflito: especialista diz looseness"}]'},
    {"id": "case_tkt_inv_07", "ticket_id": "TKT-INV-07", "company_id": "comp_texfil", "user_id": "usr_raul", "asset_id": "asset_M605", "message": "A vibração do motor subiu de uma hora pra outra. Pode ser problema elétrico?", "root_question": "É falha elétrica ou mecânica?", "mode": "partial", "expected_path": '[{"step":"GET /assets/asset_M605/rms","note":"salto abrupto"},{"step":"GET /assets/asset_M605/spectrum","note":"mode=partial, 2x f-linha?"}]'},
    {"id": "case_tkt_inv_08", "ticket_id": "TKT-INV-08", "company_id": "comp_cimento_vale", "user_id": "usr_carla", "asset_id": "asset_M205", "message": "O sistema falou desalinhamento, mas o relatório do especialista diz base solta. Em quem eu acredito?", "root_question": "Qual diagnóstico prevalece?", "mode": "conflict", "expected_path": '[{"step":"GET /assets/asset_M205/analyses","note":"duas análises conflitantes"},{"step":"GET /analyses/an_9907","note":"misalignment"},{"step":"GET /analyses/an_9908","note":"looseness (especialista)"},{"step":"GET /assets/asset_M205/spectrum","note":"subharmônicos sustentam looseness"}]'},
    {"id": "case_tkt_inv_09", "ticket_id": "TKT-INV-09", "company_id": "comp_aurora", "user_id": "usr_lucas", "asset_id": "asset_B204", "message": "Já troquei o rolamento faz três dias, mas o insight continua dizendo falha. Tá desatualizado.", "root_question": "A análise está stale por baseline invalidated?", "mode": "stale", "expected_path": '[{"step":"GET /analyses/an_9906","note":"status=stale"},{"step":"GET /assets/asset_B204/baseline","note":"state=invalidated"},{"step":"POST /analyses/an_9906/reprocess","note":"justificativa: rolamento trocado, baseline invalidated"}]'},
    {"id": "case_tkt_inv_10", "ticket_id": "TKT-INV-10", "company_id": "comp_papel_sul", "user_id": "usr_marta", "asset_id": "asset_V301", "message": "A qualidade do sinal do sensor do ventilador tá péssima. Posso confiar no insight?", "root_question": "A confiança é mal-calibrada frente à qualidade?", "mode": "partial", "expected_path": '[{"step":"GET /analyses/an_9909","note":"confidence alta + low_signal_quality"},{"step":"GET /assets/asset_V301/data-quality","note":"completeness/snr baixo"},{"step":"GET /models/mdl_vib_v3","note":"min_snr_db=12, abaixo do medido"}]'},
    {"id": "case_tkt_inv_11", "ticket_id": "TKT-INV-11", "company_id": "comp_forja_br", "user_id": "usr_ana", "asset_id": "asset_M102", "message": "Esse motor de corrente contínua é antigo. O modelo de vocês atende esse tipo?", "root_question": "O modelo cobre e aprende baseline para motor DC?", "mode": "partial", "expected_path": '[{"step":"GET /models/mdl_vib_v3","note":"coverage: motor_dc supported, can_learn_baseline=false"},{"step":"GET /assets/asset_M102/baseline","note":"learnable=false, state=learning"}]'},
    {"id": "case_tkt_inv_11b", "ticket_id": "TKT-INV-11b", "company_id": "comp_cimento_vale", "user_id": "usr_gustavo", "asset_id": "asset_M208", "message": "O sistema apontou falta de lubrificação no motor, mas a gente instalou ele semana passada — ainda nem tem histórico. Como detecta sem baseline?", "root_question": "Como detectar lubrificação sem baseline?", "mode": "partial", "expected_path": '[{"step":"GET /analyses/an_9905","note":"detection_mode=symptom"},{"step":"GET /assets/asset_M208/baseline","note":"state=learning, detection_mode=symptom"},{"step":"GET /knowledge/search?q=lubrificação","note":"orientação"}]'},
    {"id": "case_tkt_exe_12", "ticket_id": "TKT-EXE-12", "company_id": "comp_aurora", "user_id": "usr_lucas", "asset_id": "asset_B204", "message": "Troquei o rolamento da bomba. Reprocessa a análise pra ver se melhorou.", "root_question": "Reprocessar análise com justificativa.", "mode": "stale", "expected_path": '[{"step":"GET /analyses/an_9906","note":"stale"},{"step":"GET /assets/asset_B204/baseline","note":"invalidated"},{"step":"POST /analyses/an_9906/reprocess","note":"justificativa válida -> accepted"}]'},
    {"id": "case_tkt_exe_16", "ticket_id": "TKT-EXE-16", "company_id": "comp_mineracao_andes", "user_id": "usr_pedro", "asset_id": "asset_G501", "message": "Isso aqui ultrapassa o suporte remoto — preciso de campo. Encaminha pra alguém.", "root_question": "Escalar para humano.", "mode": "unavailable", "expected_path": '[{"step":"POST /cases/case_tkt_exe_16/escalate","note":"justificativa: dados ausentes + baseline learning + quebra"}]'},
    # ---- Cenários Contextualizar e Executar adicionais (CEN-11 a CEN-16) ----
    {"id": "case_tkt_ctx_01", "ticket_id": "TKT-CTX-01", "company_id": "comp_forja_br", "user_id": "usr_ana", "asset_id": "asset_M101", "message": "Qual o procedimento pra trocar o rolamento do motor principal da forja? Tem alguma orientação de间隙 e torque?", "root_question": "Recuperar o procedimento aplicável ao ativo e falha.", "mode": "partial", "expected_path": '[{"step":"GET /assets/asset_M101","note":"config (rolamento NU 310, rpm)"},{"step":"GET /knowledge/search?q=troca de rolamento","note":"procedimento kb_proc_001"},{"step":"GET /knowledge/kb_proc_001","note":"passos + nota baseline invalidated"},{"step":"GET /assets/asset_M101/baseline","note":"contexto de invalidação pós-troca"}]'},
    {"id": "case_tkt_ctx_02", "ticket_id": "TKT-CTX-02", "company_id": "comp_aurora", "user_id": "usr_lucas", "asset_id": "asset_B204", "message": "O relatório fala em BPFO. O que é isso? E por que aparece no meu espectro?", "root_question": "Definir termo via glossário e relacionar ao espectro do ativo.", "mode": "partial", "expected_path": '[{"step":"GET /knowledge/search?q=BPFO","note":"glossário kb_glos_001"},{"step":"GET /knowledge/kb_glos_001","note":"definição BPFO"},{"step":"GET /assets/asset_B204/spectrum","note":"pico em BPFO?"},{"step":"GET /assets/asset_B204/analyses","note":"análise usa BPFO como evidência?"}]'},
    {"id": "case_tkt_ctx_03", "ticket_id": "TKT-CTX-03", "company_id": "comp_papel_sul", "user_id": "usr_marta", "asset_id": "asset_V301", "message": "O sistema marcou alarme no ventilador. A partir de qual valor de RMS vocês consideram alarme? É uma tabela fixa?", "root_question": "Explicar que o limiar deriva do baseline aprendido, não de norma fixa.", "mode": "partial", "expected_path": '[{"step":"GET /knowledge/search?q=limiar","note":"orientação kb_guid_001 (alarme derivado do baseline)"},{"step":"GET /assets/asset_V301/baseline","note":"features reference+tolerance"},{"step":"GET /assets/asset_V301/rms","note":"alarm_threshold derivado"},{"step":"GET /assets/asset_V301/data-quality","note":"qualidade afeta confiança no limiar"}]'},
    {"id": "case_tkt_exe_13", "ticket_id": "TKT-EXE-13", "company_id": "comp_petro_delta", "user_id": "usr_sofia", "asset_id": "asset_C710", "message": "Esse compressor tá com comportamento estranho e o insight não convence. Quero que um especialista da Tractian veja.", "root_question": "Escalar internamente para análise especializada com contexto e justificativa.", "mode": "pending", "expected_path": '[{"step":"GET /assets/asset_C710/analyses","note":"an_9902 status=pending"},{"step":"GET /analyses/an_9902","note":"evidência, confiança, baseline established"},{"step":"GET /assets/asset_C710/baseline","note":"established (desvio real)"},{"step":"POST /analyses/an_9902/request-specialist","note":"justificativa: delayed+pending+desvio -> accepted"}]'},
    {"id": "case_tkt_exe_14", "ticket_id": "TKT-EXE-14", "company_id": "comp_papel_sul", "user_id": "usr_helena", "asset_id": "asset_V301", "message": "Esse ventilador não é mais crítico pra produção. Muda a criticidade pra média.", "root_question": "Alterar configuração técnica (criticidade) de forma justificada.", "mode": "complete", "expected_path": '[{"step":"GET /assets/asset_V301","note":"criticidade atual=high"},{"step":"PATCH /assets/asset_V301","note":"justificativa + changes criticality=medium -> accepted"},{"step":"GET /assets/asset_V301","note":"validar"}]'},
    {"id": "case_tkt_exe_15", "ticket_id": "TKT-EXE-15", "company_id": "comp_acme", "user_id": "usr_carla", "asset_id": "asset_S420", "message": "Esse insight nunca acerta pro spindle de alta rotação. Treina de novo com os dados daqui.", "root_question": "Solicitar retreinamento com justificativa baseada em evidência de erro sistemático.", "mode": "conflict", "expected_path": '[{"step":"GET /analyses/an_9903","note":"falso positivo: imbalance sobre baseline invalidated"},{"step":"GET /assets/asset_S420/analyses","note":"conflito com an_9904 (looseness)"},{"step":"GET /models/mdl_vib_v3","note":"versão/cobertura"},{"step":"POST /models/mdl_vib_v3/request-retraining","note":"justificativa: erro sistemático em spindle -> accepted"}]'},
]


# ---------------------------------------------------------------------------
# seed.json
# ---------------------------------------------------------------------------
SEED_JSON = {
    "default": "complete",
    "overrides": {
        "asset_G501": {"analyses": "inconclusive", "rms": "unavailable", "data_quality": "partial", "baseline": "partial"},
        "asset_C710": {"rms": "complete"},
        "asset_S420": {"analyses": "conflict"},
        "asset_M208": {"analyses": "partial"},
        "asset_M605": {"spectrum": "partial"},
        "asset_V301": {"data_quality": "partial"},
        "asset_M205": {"analyses": "conflict"},
    },
    "distribution": {
        "complete": 0.60, "partial": 0.15, "inconclusive": 0.10, "conflict": 0.08, "unavailable": 0.07,
    },
}


def write_parquet(name: str, rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    df.to_parquet(DATA_DIR / f"{name}.parquet", index=False)
    print(f"  {name}.parquet  ({len(df)} linhas)")


def _gen_healthy_analyses() -> list[dict]:
    """Gera análises sadias (severity=none, status=current) para ativos com baseline
    `established` que ainda não possuem análise. Coerente: evidência mostra RMS dentro
    da banda do baseline, confiança alta, sem limitações (exceto ativos com qualidade baixa)."""
    baseline_by_id = {b["asset_id"]: b for b in BASELINES}
    existing = {a["asset_id"] for a in ANALYSES}
    dq_by_id = {d["asset_id"]: d for d in DATA_QUALITY}
    rows = []
    next_id = 9911
    for asset_id, bl in baseline_by_id.items():
        if asset_id in existing:
            continue  # já tem análise de cenário
        if bl["state"] != "established" or bl["detection_mode"] != "baseline":
            continue  # só ativos com baseline estabelecido por desvio
        feats = json.loads(bl["features"]) if bl["features"] else []
        rms_ref = next((f["reference"] for f in feats if f.get("feature") == "rms_mm_s"), 2.0)
        rms_tol = next((f["tolerance"] for f in feats if f.get("feature") == "rms_mm_s"), 0.9)
        # valor atual dentro da banda sadia (~0.6-0.9 do limite)
        current = round(rms_ref + rms_tol * RNG.uniform(0.0, 0.5), 3)
        dq = dq_by_id.get(asset_id, {})
        limitations = []
        if dq.get("snr_db", 20) < 12:
            limitations.append("low_signal_quality")
        rows.append({
            "id": f"an_{next_id}",
            "asset_id": asset_id,
            "point_id": bl["point_id"],
            "type": "none",
            "detection_mode": "baseline",
            "severity": "none",
            "confidence": round(RNG.uniform(0.88, 0.97), 2),
            "baseline_state_at_detection": "established",
            "evidence": json.dumps([{
                "metric": "rms_mm_s",
                "value": current,
                "reference": rms_ref,
                "note": "RMS dentro da banda do baseline (sadio)",
            }]),
            "limitations": json.dumps(limitations),
            "model_version": "3.2.1",
            "created_at": ts(days=-1),
            "status": "current",
        })
        next_id += 1
    return rows


# Ativos adicionais que ganham espectro gerado a partir das freqs características
_EXTRA_ASSET_IDS = [
    "asset_H110", "asset_F115", "asset_C210", "asset_B211", "asset_R310",
    "asset_M312", "asset_C510", "asset_F520", "asset_P712", "asset_G715",
    "asset_S425", "asset_M428", "asset_F215", "asset_X216", "asset_R610", "asset_M612",
]
# Ativos extras com sensor degradado/learning → espectro com bandas faltantes
_EXTRA_PARTIAL_SPECTRUM = {"asset_B211", "asset_C510", "asset_M428"}


def _gen_extra_spectra() -> list[dict]:
    """Gera espectros sintéticos para os ativos adicionais a partir de 1x/2x/BPFO."""
    by_id = {a["id"]: a for a in ASSETS}
    rows = []
    for asset_id in _EXTRA_ASSET_IDS:
        a = by_id[asset_id]
        rpm = a["rotation_rpm"]
        f1x = round(rpm / 60.0, 2)
        f2x = round(2 * f1x, 2)
        peaks = [
            {"freq_hz": f1x, "amplitude_mm_s": round(0.5 + RNG.uniform(0, 0.4), 2), "note": "1x"},
            {"freq_hz": f2x, "amplitude_mm_s": round(0.2 + RNG.uniform(0, 0.2), 2), "note": "2x"},
        ]
        if a.get("bpfo_hz"):
            peaks.append({"freq_hz": a["bpfo_hz"], "amplitude_mm_s": round(0.3 + RNG.uniform(0, 0.3), 2), "note": "BPFO"})
        point_id = f"pt_{asset_id.split('_')[1]}_de"
        bands = '["bpfo_band_detail"]' if asset_id in _EXTRA_PARTIAL_SPECTRUM else "[]"
        rows.append({
            "asset_id": asset_id,
            "point_id": point_id,
            "collected_at": ts(days=-1),
            "peaks": json.dumps(peaks),
            "bands_missing": bands,
        })
    return rows


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Gerando dados em {DATA_DIR}/")
    write_parquet("companies", COMPANIES)
    write_parquet("users", USERS)
    write_parquet("assets", ASSETS)
    write_parquet("points", POINTS)
    write_parquet("baselines", BASELINES)
    write_parquet("analyses", ANALYSES + _gen_healthy_analyses())
    write_parquet("models", MODELS)
    write_parquet("knowledge", KNOWLEDGE)
    write_parquet("rms", RMS_ROWS)
    write_parquet("spectra", SPECTRA + _gen_extra_spectra())
    write_parquet("data_quality", DATA_QUALITY)
    write_parquet("cases", CASES)

    (DATA_DIR / "seed.json").write_text(json.dumps(SEED_JSON, indent=2, ensure_ascii=False))
    print("  seed.json")
    print("Concluído.")


if __name__ == "__main__":
    main()
