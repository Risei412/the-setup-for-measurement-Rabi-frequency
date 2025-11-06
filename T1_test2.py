# -*- coding: utf-8 -*-
"""
T1 measurement (variable dark time τ) using:
  – Pulse-Blaster (SpinAPI) for timing
  – NI USB-6363 counter with Pause-Trigger gating
"""

import time
import numpy as np
import nidaqmx
from nidaqmx.constants import Edge, Level
from spinapi import *
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("TkAgg")  # TkAggバックエンドを使用
import matplotlib.pyplot as plt

# --------------------------- USER  PARAMETERS ---------------------------
DEV          = "Dev1"       # NI デバイス名
SPCM_PFI     = 0            # SPCM → /Dev1/PFI0
GATE_PFI     = 1            # Gate → /Dev1/PFI1
CTR          = 2            # 使うカウンタ ctr2

AOM = 1
GATE = 2

PREP_NS      = 3_000        # 初期化レーザ   幅 [ns]  (3 µs)
READ_NS      = 30000        # 読み出しレーザ 幅 [ns]  (3 µs)
COOLDOWN_NS  = 5_000        # クールダウン   幅 [ns]
LEAD_AOM_NS = 500
CORE_CLK_MHZ = 500.0        # Pulse-Blaster core clock

TAU_US_LIST  = np.geomspace(0.1, 1000, 10)  # τ [µs] を 0.1–5000 µs で対数掃引
N_AVERAGE    = 10        # 各 τ でのショット回数
# ------------------------------------------------------------------------

# --- convenience --------------------------------------------------------
def ns(ns_val): return ns_val          # alias for readability
def us(us_val): return us_val * 1_000
# ------------------------------------------------------------------------

# ============ 1.  Pulse-Blaster 初期化 (1 回だけ) =======================
pb_select_board(0)
pb_init()
pb_core_clock(CORE_CLK_MHZ)            # 500 MHz → 2 ns resolution

def program_sequence(tau_us):
    """tau_us [µs] を受け取り、AOM 遅延 500 ns を織り込んだ PB シーケンスを再コンパイル"""
    tau_ns = us(tau_us)

    pb_start_programming(PULSE_PROGRAM)

    # 0) 同期ウェイト（100 ns）
    pb_inst_pbonly(0, CONTINUE, 0, ns(100))

    # 1) 初期化レーザ（Gate は閉じたまま）
    pb_inst_pbonly(1<<AOM, CONTINUE, 0, PREP_NS)
    
    pb_inst_pbonly(0, CONTINUE, 0, LEAD_AOM_NS)

    # 2) 暗黒期間 τ
    pb_inst_pbonly(0, CONTINUE, 0, tau_ns)

    # 3-a) 読み出し：AOM を先に High、500 ns 待つ
    pb_inst_pbonly(1<<AOM, CONTINUE, 0, LEAD_AOM_NS)

    # 3-b) Gate も High にして 〝計数窓 READ_NS〟
    pb_inst_pbonly((1<<AOM)|(1<<GATE),
                   CONTINUE, 0, READ_NS)

    # 3-c) Gate を閉じてから AOM を 500 ns 引きずる（任意）
    # pb_inst_pbonly(1<<AOM, CONTINUE, 0, TRAIL_AOM_NS)

    # 4) クールダウン
    pb_inst_pbonly(0, CONTINUE, 0, COOLDOWN_NS)

    # 5) STOP
    pb_inst_pbonly(0, STOP, 0, ns(100))

    pb_stop_programming()

# ============ 2.  NI-DAQmx タスク（Pause-Trigger 方式） ================
def create_counter_task():
    task = nidaqmx.Task()
    ch = task.ci_channels.add_ci_count_edges_chan(
            counter=f"{DEV}/ctr{CTR}",
            edge=Edge.RISING,
            initial_count=0)
    ch.ci_count_edges_term = f"/{DEV}/PFI{SPCM_PFI}"   # SPCM
    pt = task.triggers.pause_trigger
    pt.trig_type     = nidaqmx.constants.TriggerType.DIGITAL_LEVEL
    pt.dig_lvl_src   = f"/{DEV}/PFI{GATE_PFI}"         # Gate
    pt.dig_lvl_when  = Level.HIGH                       # Low で停止 → High 中だけ数える
    return task

# ============ 3.  ループ測定 ===========================================
results = []

with create_counter_task() as task:
    for tau in TAU_US_LIST:
        program_sequence(tau)         # PB に τ ごとのシーケンスを書き込み
        counts_sum = 0
        task.start()   
        for _ in range(N_AVERAGE):
                       # Arm
            pb_start()                # ショット実行
            # --- 完了待ち（プログラム長分 + 余裕） ---
            prog_len_s = (PREP_NS + LEAD_AOM_NS + us(tau) + READ_NS + COOLDOWN_NS + 200) * 1e-9
            time.sleep(prog_len_s + 0.0001)  # 1 ms 余裕
                   # Pause-Trig が閉じた後なので即読みOK
            
        result = task.read()       # Pause-Trig が閉じた後なので即読みOK
        task.stop()
        # mean_counts = counts_sum / N_AVERAGE
        # total_counts = counts_sum   
        results.append(result)
        print(f"τ = {tau:7.2f} µs  →  ⟨counts⟩ = {result:.1f}")
pb_close()


tau_arr    = np.asarray(TAU_US_LIST)      # τ 配列  [µs]
counts_arr = np.asarray(results)          # 合計フォトン数 Σcounts(τ)

# ---------- 4.  T1 フィット（合計カウント版） -----------------
def exp_decay_sum(t, A_sum, T1, B_sum):
    """A_sum = 単発振幅×N_ave,  B_sum = 単発BG×N_ave"""
    return A_sum * np.exp(-t / T1) + B_sum

# 初期値：max − min を振幅、mean を T1、min を BG とする
p0 = [counts_arr.max() - counts_arr.min(), tau_arr.mean(), counts_arr.min()]

# Poisson 誤差（√N）で重みづけするとフィットが安定
sigma = np.sqrt(np.clip(counts_arr, 1, None))       # 0 除け
pars, pcov = curve_fit(exp_decay_sum,
                       tau_arr, counts_arr,
                       p0=p0, sigma=sigma,
                       absolute_sigma=True)

A_sum_fit, T1_fit, B_sum_fit = pars
print(f"\n推定 T1 ≈ {T1_fit:,.1f} µs")

# ---------- 5.  プロット（軸ラベルも合計カウントに変更） ------------
plt.semilogx(tau_arr, counts_arr, "o", label="data (Σcounts)")
t_fit = np.logspace(np.log10(tau_arr.min()), np.log10(tau_arr.max()), 500)
plt.semilogx(t_fit, exp_decay_sum(t_fit, *pars),
             "-", label=f"fit  T1={T1_fit:.0f} µs")
plt.xlabel("dark time τ (µs)")
plt.ylabel("total photon counts per τ")
plt.legend()
plt.tight_layout()
plt.show()

