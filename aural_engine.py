"""
VO-SE Pro - AuralAI Model Trainer（ミニマル版）

必要なものはこれだけ:
  - 単独音WAV（例: あ.wav, か.wav）
  - oto.ini

使い方:
  1. PHONEMES に学習したい音素を書く（デフォルトは「あ」「か」）
  2. assets/training_voices/ に音源フォルダを置く
  3. python train_aural_model.py
  4. models/aural_dynamics.onnx が生成される

pip install numpy scikit-learn skl2onnx onnxruntime
(任意) pip install scipy librosa
"""

import os
import glob
import wave
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

# ─────────────────────────────────────────────
# ★ ここだけ編集すればOK
# ─────────────────────────────────────────────
PHONEMES = ["あ", "か"]   # 学習する音素リスト。増やす場合はここに追加するだけ

VOICE_DIR   = "assets/training_voices"
OUTPUT_PATH = "models/aural_dynamics.onnx"
TARGET_FS   = 44100
N_MFCC      = 20
N_FRAMES    = 64
MAX_MS      = 300.0  # ラベル正規化の上限(ms)

# ─────────────────────────────────────────────
# PHONEMES から自動生成（編集不要）
# ─────────────────────────────────────────────
PHONEME_INDEX: Dict[str, int] = {p: i for i, p in enumerate(PHONEMES)}
N_PHONEME = len(PHONEMES)
MFCC_DIM  = N_MFCC * N_FRAMES
INPUT_DIM = MFCC_DIM + N_PHONEME  # 特徴量の総次元数


# ─────────────────────────────────────────────
# 1. oto.ini パーサー
# ─────────────────────────────────────────────
@dataclass
class OtoEntry:
    wav_path:      str
    alias:         str
    offset:        float
    consonant:     float
    cutoff:        float
    pre_utterance: float
    overlap:       float


def parse_oto_ini(oto_path: str) -> List[OtoEntry]:
    entries: List[OtoEntry] = []
    voice_dir = os.path.dirname(oto_path)

    for enc in ("cp932", "utf-8", "utf-8-sig"):
        try:
            with open(oto_path, encoding=enc) as f:
                lines = f.readlines()
            break
        except (UnicodeDecodeError, FileNotFoundError):
            continue
    else:
        print(f"  [WARN] 読めません: {oto_path}")
        return entries

    seen: set = set()
    for line in lines:
        line = line.strip()
        if not line or line.startswith(";") or "=" not in line:
            continue
        try:
            wav_name, params_str = line.split("=", 1)
            wav_name = wav_name.strip()
            parts = params_str.split(",")
            if len(parts) < 5:
                continue

            alias = parts[0].strip() or Path(wav_name).stem

            # PHONEMESに含まれない音素はスキップ
            if alias not in PHONEME_INDEX:
                continue

            wav_path = os.path.join(voice_dir, wav_name)
            if not wav_path.lower().endswith(".wav"):
                wav_path += ".wav"

            # 1WAV=1エントリ保証
            if wav_path in seen:
                continue
            seen.add(wav_path)

            entries.append(OtoEntry(
                wav_path      = wav_path,
                alias         = alias,
                offset        = float(parts[1]) if parts[1].strip() else 0.0,
                consonant     = float(parts[2]) if parts[2].strip() else 0.0,
                cutoff        = float(parts[3]) if parts[3].strip() else 0.0,
                pre_utterance = float(parts[4]) if parts[4].strip() else 0.0,
                overlap       = float(parts[5]) if len(parts) > 5 and parts[5].strip() else 0.0,
            ))
        except (ValueError, IndexError):
            continue

    return entries


# ─────────────────────────────────────────────
# 2. WAV読み込み
# ─────────────────────────────────────────────
def load_wav(path: str) -> Optional[np.ndarray]:
    try:
        with wave.open(path, "rb") as wf:
            fs      = wf.getframerate()
            nch     = wf.getnchannels()
            raw     = wf.readframes(wf.getnframes())

        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if nch == 2:
            data = data[::2]

        if fs != TARGET_FS:
            from math import gcd
            g = gcd(fs, TARGET_FS)
            try:
                from scipy.signal import resample_poly
                data = resample_poly(data, TARGET_FS // g, fs // g).astype(np.float32)
            except ImportError:
                n_new = int(len(data) * TARGET_FS / fs)
                data = np.interp(
                    np.linspace(0, len(data) - 1, n_new),
                    np.arange(len(data)), data
                ).astype(np.float32)

        return data
    except Exception:
        return None


# ─────────────────────────────────────────────
# 3. MFCC抽出
# ─────────────────────────────────────────────
def extract_mfcc(signal: np.ndarray) -> np.ndarray:
    try:
        import librosa
        return librosa.feature.mfcc(y=signal, sr=TARGET_FS, n_mfcc=N_MFCC)
    except ImportError:
        pass

    # librosa なし: 純NumPy版
    n_fft, hop, n_mels = 512, 256, 40
    window = np.hanning(n_fft)
    frames = [np.abs(np.fft.rfft(signal[i:i+n_fft] * window))
              for i in range(0, len(signal) - n_fft, hop)]
    if not frames:
        return np.zeros((N_MFCC, 1))
    spec = np.array(frames).T

    f_min, f_max = 0.0, TARGET_FS / 2.0
    mel_pts  = np.linspace(2595*np.log10(1+f_min/700),
                           2595*np.log10(1+f_max/700), n_mels+2)
    freq_pts = 700*(10**(mel_pts/2595)-1)
    bins     = np.floor((n_fft+1)*freq_pts/TARGET_FS).astype(int)

    fbank = np.zeros((n_mels, spec.shape[0]))
    for m in range(1, n_mels+1):
        for k in range(bins[m-1], bins[m]):
            if bins[m] != bins[m-1]:
                fbank[m-1, k] = (k-bins[m-1])/(bins[m]-bins[m-1])
        for k in range(bins[m], bins[m+1]):
            if bins[m+1] != bins[m]:
                fbank[m-1, k] = (bins[m+1]-k)/(bins[m+1]-bins[m])

    log_mel = np.log(np.dot(fbank, spec) + 1e-8)
    dct = np.cos(np.pi*np.outer(np.arange(N_MFCC),
                                np.arange(1, n_mels+1)-0.5)/n_mels)
    return np.dot(dct, log_mel)


def mfcc_to_fixed(mfcc: np.ndarray) -> np.ndarray:
    if mfcc.shape[1] >= N_FRAMES:
        mfcc = mfcc[:, :N_FRAMES]
    else:
        mfcc = np.hstack([mfcc, np.zeros((mfcc.shape[0], N_FRAMES-mfcc.shape[1]))])
    return mfcc.flatten().astype(np.float32)


# ─────────────────────────────────────────────
# 4. 音素 one-hot
# ─────────────────────────────────────────────
def to_onehot(alias: str) -> np.ndarray:
    vec = np.zeros(N_PHONEME, dtype=np.float32)
    if alias in PHONEME_INDEX:
        vec[PHONEME_INDEX[alias]] = 1.0
    return vec


# ─────────────────────────────────────────────
# 5. 教師データ生成
# ─────────────────────────────────────────────
def build_dataset(voice_root: str) -> Tuple[np.ndarray, np.ndarray]:
    oto_files = glob.glob(os.path.join(voice_root, "**", "oto.ini"), recursive=True)
    print(f"oto.ini: {len(oto_files)}件")
    print(f"対象音素: {PHONEMES}")

    X_list, y_list = [], []
    skipped = 0

    for oto_path in oto_files:
        entries = parse_oto_ini(oto_path)
        if entries:
            print(f"  {Path(oto_path).parent.name}: {len(entries)}件マッチ")

        for e in entries:
            if not os.path.exists(e.wav_path):
                skipped += 1
                continue

            sig = load_wav(e.wav_path)
            if sig is None or len(sig) < TARGET_FS * 0.03:
                skipped += 1
                continue

            start   = max(0, int(e.offset * TARGET_FS / 1000))
            segment = sig[start:]
            if len(segment) < 256:
                skipped += 1
                continue

            feat = np.concatenate([
                mfcc_to_fixed(extract_mfcc(segment)),  # MFCC
                to_onehot(e.alias),                    # one-hot
            ])

            label = np.array([
                np.clip(e.pre_utterance / MAX_MS, 0.0, 1.0),
                np.clip(e.overlap       / MAX_MS, 0.0, 1.0),
                np.clip(e.consonant     / MAX_MS, 0.0, 1.0),
            ], dtype=np.float32)

            X_list.append(feat)
            y_list.append(label)

    print(f"\n有効サンプル: {len(X_list)}件  スキップ: {skipped}件")

    if not X_list:
        raise RuntimeError(
            f"サンプルが0件です。\n"
            f"VOICE_DIR={voice_root} にoto.iniとWAVがあるか確認してください。\n"
            f"対象音素: {PHONEMES}"
        )

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


# ─────────────────────────────────────────────
# 6. 学習
# ─────────────────────────────────────────────
def train(X: np.ndarray, y: np.ndarray):
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split

    print(f"\n学習開始: {len(X)}サンプル, 入力{X.shape[1]}次元")

    if len(X) >= 20:
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.15, random_state=42)
    else:
        X_tr, X_val, y_tr, y_val = X, X, y, y
        print("  サンプルが少ないため検証セットなしで学習します")

    if len(X_tr) >= 10:
        from sklearn.neural_network import MLPRegressor
        # 隠れ層を 256-128-64 の三層構成で明示
        print("  モデル: MLP (256-128-64) 三層学習構成")
        model = Pipeline([
            ("sc",  StandardScaler()),
            ("mlp", MLPRegressor(
                hidden_layer_sizes=(256, 128, 64),
                activation="relu",
                solver="adam",
                alpha=0.0001,
                max_iter=1000,
                learning_rate_init=0.001,
                early_stopping=len(X_tr) >= 20,
                n_iter_no_change=20,
                verbose=True,
                random_state=42,
            ))
        ])
    else:
        from sklearn.linear_model import Ridge
        print("  モデル: Ridge（サンプル不足のため簡易モデル）")
        model = Pipeline([("sc", StandardScaler()), ("r", Ridge())])

    model.fit(X_tr, y_tr)
    print(f"  検証R²: {model.score(X_val, y_val):.4f}")
    return model


# ─────────────────────────────────────────────
# 7. ONNXエクスポート
# ─────────────────────────────────────────────
def export_onnx(model, path: str):
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType

        onnx_model = convert_sklearn(
            model, initial_types=[("input", FloatTensorType([None, INPUT_DIM]))]
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        print(f"\n保存: {path}")
        print(f"  入力: (batch, {INPUT_DIM})  出力: (batch, 3)")

    except ImportError:
        import pickle
        pkl = path.replace(".onnx", ".pkl")
        with open(pkl, "wb") as f:
            pickle.dump(model, f)
        print(f"skl2onnx未インストール → {pkl} に保存しました")
        print("pip install skl2onnx でONNX出力できます")


# ─────────────────────────────────────────────
# 8. 動作確認
# ─────────────────────────────────────────────
def verify(path: str):
    try:
        import onnxruntime as ort
        sess  = ort.InferenceSession(path)
        dummy = np.concatenate([
            np.zeros((1, MFCC_DIM), dtype=np.float32),
            to_onehot(PHONEMES[0]).reshape(1, -1),
        ], axis=1)
        out = sess.run(None, {"input": dummy})[0]
        print(f"\n動作確認OK (音素='{PHONEMES[0]}' ダミー入力):")
        print(f"  pre_utterance: {out[0][0]*MAX_MS:.1f}ms")
        print(f"  overlap      : {out[0][1]*MAX_MS:.1f}ms")
        print(f"  consonant    : {out[0][2]*MAX_MS:.1f}ms")
    except Exception as e:
        print(f"動作確認スキップ: {e}")


# ─────────────────────────────────────────────
# 9. ダミーデータ（音源ゼロ時）
# ─────────────────────────────────────────────
def dummy_dataset() -> Tuple[np.ndarray, np.ndarray]:
    print(f"\n[DEMOモード] {VOICE_DIR} に音源が見つかりません。")
    print(f"対象音素 {PHONEMES} のダミーデータで動作確認します。\n")

    # 音素ごとの典型値（ms）
    defaults = {
        "あ": (80, 40, 30),
        "か": (90, 45, 60),
        "い": (75, 35, 28),
        "う": (78, 38, 25),
        "さ": (95, 48, 70),
    }
    rng = np.random.default_rng(42)
    X_list, y_list = [], []

    for p in PHONEMES:
        pre, ov, con = defaults.get(p, (80, 40, 40))
        for _ in range(30):
            feat = np.concatenate([
                rng.standard_normal(MFCC_DIM).astype(np.float32),
                to_onehot(p),
            ])
            label = np.array([
                np.clip(rng.normal(pre, 5) / MAX_MS, 0.0, 1.0),
                np.clip(rng.normal(ov,  5) / MAX_MS, 0.0, 1.0),
                np.clip(rng.normal(con, 5) / MAX_MS, 0.0, 1.0),
            ], dtype=np.float32)
            X_list.append(feat)
            y_list.append(label)

    print(f"ダミーデータ: {len(X_list)}サンプル ({len(PHONEMES)}音素 × 30)")
    return np.array(X_list), np.array(y_list)


# ─────────────────────────────────────────────
# メイン
# ─────────────────────────────────────────────
def main():
    print("=" * 50)
    print("  VO-SE Pro - AuralAI Trainer")
    print(f"  対象音素: {PHONEMES}")
    print("=" * 50)

    has_voice = (
        os.path.exists(VOICE_DIR)
        and glob.glob(os.path.join(VOICE_DIR, "**", "oto.ini"), recursive=True)
    )

    X, y = build_dataset(VOICE_DIR) if has_voice else dummy_dataset()

    print(f"\nデータ統計 (×{MAX_MS:.0f}=ms):")
    print(f"  pre_utterance  平均{y[:,0].mean()*MAX_MS:.1f}ms")
    print(f"  overlap        平均{y[:,1].mean()*MAX_MS:.1f}ms")
    print(f"  consonant      平均{y[:,2].mean()*MAX_MS:.1f}ms")

    model = train(X, y)
    export_onnx(model, OUTPUT_PATH)

    if os.path.exists(OUTPUT_PATH):
        verify(OUTPUT_PATH)

    print("\n" + "=" * 50)
    print("  完了!")
    print("=" * 50)


if __name__ == "__main__":
    main()
