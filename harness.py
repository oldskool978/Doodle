from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

sys.modules["torchvision"] = None
sys.modules["torchvision.transforms"] = None
sys.modules["torchvision.io"] = None

warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")

import numpy as np

from engine import MusicEngine
from schema import (
    DEFAULT_PRESET_FILENAME,
    GenerationRequest,
    GenerationResponse,
    SUPPORTED_COT,
    SUPPORTED_ODE_METHODS,
    get_active_engine_defaults,
    has_custom_default_preset,
)

DEFAULT_HARNESS_VOCAL_LYRICS = """[intro]
(Smooth Rhodes chords, filtered 808 glide, ad-libs)
Yeah, listen
Midnight in the city, let the groove breathe
Oh, oh-woah, yeah

[verse 1]
Midnight riding under neon streetlights
Searching for the answers in the rearview mirror
Thought I had the blueprint solid in my mind
Now the silhouette of you is drawing nearer
Dashboard glowing with a steady slow pulse
Echoes of your whisper in the night air

[pre-chorus 1]
I try to fight it, but it's pulling me in
Every harmonic frequency starts spinning again
Tension rising from the bottom to top
Got that momentum and we never gon' stop

[chorus 1]
Got me caught up in the way that you move
Nobody else can lock right into the groove
Got my heart on the floor, baby, give me one more
Show me that rhythm, tell me what you wanna do
(Yeah, yeah, keep it right there)

[verse 2]
Two in the morning, baseline taking over
Sip of something smooth, leaning in a little closer
Sub-frequencies vibrating the floor
You give me everything, but I still want more
Syncopated touch, perfect timing on the beat
Fire in our eyes, generating pure heat

[pre-chorus 2]
I try to fight it, but it's pulling me in
Every harmonic frequency starts spinning again
Tension rising from the bottom to top
Got that momentum and we never gon' stop

[chorus 2]
Got me caught up in the way that you move
Nobody else can lock right into the groove
Got my heart on the floor, baby, give me one more
Show me that rhythm, tell me what you wanna do
(Yeah, yeah, right into the pocket)

[bridge]
Take it to the falsetto high, let the bass drop clean
Smoothest vibration that you've ever seen
Counterpoint melodies weaving around
Elevating the pressure, capturing the sound
Hold that note, let the energy soar
Take it to places that we never went before

[solo]
(Warm expressive nylon and electric guitar soloing over deep sub-bass and syncopated percussion)

[chorus 3]
Got me caught up in the way that you move
Nobody else can lock right into the groove
Got my heart on the floor, baby, give me one more
Show me that rhythm, tell me what you wanna do
(Oh-woah, give me one more time)

[outro]
Fade into the low-end frequency
Keep the drum pocket steady for me
Ad-libs drifting out into the night
Yeah, just like that
Fade to black"""

DEFAULT_HARNESS_INSTRUMENTAL_CUES = """[intro]
(Warm Fender Rhodes chords, vinyl crackle, subtle tape delay)

[theme a]
(Sub-bass 808 glides, syncopated rimshot, closed hi-hats)

[verse 1]
(Acoustic nylon guitar arpeggios, melodic motif, steady groove)

[pre-chorus 1]
(Rising synth pad swells, filtered white noise sweeps, building tension)

[chorus 1]
(Punchy kick drum, melodic lead synth, stereo chorus, dynamic claps)

[verse 2]
(Stripped drum pocket, expressive legato electric guitar, chord changes)

[pre-chorus 2]
(Rolling 32nd-note hi-hat accents, sharp brass stabs, rising crescendo)

[chorus 2]
(Climactic full rhythm section, driving 808, soaring harmonic layers)

[bridge]
(Half-time breakdown, filtered Rhodes chords, resonant sub drops)

[solo]
(Overdriven electric guitar solo, dynamic pitch slides, legato phrasing)

[chorus 3]
(Final explosive climax, layered counter-melodies, maximum punch)

[outro]
(Solitary Rhodes chords, decaying reverb tails, low-end filter fade)"""


def create_default_harness_request() -> GenerationRequest:
    defaults = get_active_engine_defaults()
    default_vocal = "Silky male tenor lead vocal, dynamic chest-to-falsetto transitions, intricate melismatic ad-libs, stacked 4-part harmonies."
    default_inst = "Warm Fender Rhodes chords, expressive legato nylon guitar, and melodic synthesizer leads."
    return GenerationRequest(
        genre="Contemporary R&B",
        subgenre="2000s Pop R&B / Slow Jam Bounce",
        bpm=96,
        key="F minor",
        mood="Sensual, passionate, smooth, confident, driving.",
        vocals=default_vocal,
        vocal_lead=default_vocal,
        instrumental_lead=default_inst,
        arrangement="Primary: Warm Fender Rhodes chords and expressive nylon guitar arpeggios establish the core harmonic progression. Secondary: A deep sliding 808 sub-bass enters alongside crisp syncopated rimshots and 16th-note hi-hat rolls. The chorus expands with rich analog string pads and dynamic claps, while the bridge strips back to solitary Rhodes voicings before a climactic final hook.",
        lyrics=DEFAULT_HARNESS_VOCAL_LYRICS,
        instrumental_lyrics=DEFAULT_HARNESS_INSTRUMENTAL_CUES,
        is_instrumental=False,
        instrumental_branch="cues",
        cot=defaults["cot"],
        temperature=defaults["temperature"],
        top_p=defaults["top_p"],
        top_k=defaults["top_k"],
        repetition_penalty=defaults["repetition_penalty"],
        penalty_window=defaults["penalty_window"],
        abc_temperature=defaults["abc_temperature"],
        abc_top_p=defaults["abc_top_p"],
        abc_top_k=defaults["abc_top_k"],
        abc_repetition_penalty=defaults["abc_repetition_penalty"],
        abc_penalty_window=defaults["abc_penalty_window"],
        cfg_scale=defaults["cfg_scale"],
        num_inference_steps=defaults["num_inference_steps"],
        ode_method=defaults["ode_method"],
        vae_core_frames=defaults["vae_core_frames"],
        vae_halo_frames=defaults["vae_halo_frames"],
        audio_duration=240.0,
        seed=42,
        output_path="output_vocal_master.wav",
        apply_declick=defaults["apply_declick"],
        cpu_offload=defaults["cpu_offload"],
    )


def print_telemetry(resp: GenerationResponse, req: Optional[GenerationRequest] = None) -> None:
    print("\n" + "=" * 84)
    print("                        DOODLE ACOUSTIC TELEMETRY REPORT")
    print("=" * 84)
    print(f"Master Destination:    {resp.output_path}")
    print(f"Sampling Resolution:   {resp.sample_rate} Hz (32-bit Float PCM Stereo)")
    print(f"Audio Duration:        {resp.duration_seconds:.4f}s ({resp.total_samples:,} samples)")
    print(f"Inference Latency:     {resp.generation_time_seconds:.4f}s (RTF: {resp.real_time_factor:.4f}x)")
    print(f"Peak VRAM Footprint:   {resp.peak_vram_gb:.3f} GB")
    print(f"Memory Architecture:   {'SEQUENTIAL CPU OFFLOAD' if resp.cpu_offload_active else 'RESIDENT GPU VRAM'}")
    print(f"Modality Mode:         {'INSTRUMENTAL' if resp.is_instrumental_used else 'VOCAL SONG'}")
    if resp.is_instrumental_used:
        print(f"Instrumental Branch:   {resp.instrumental_branch_used.upper()}")
    print(f"Chain-of-Thought Plan: {resp.cot_mode_used.upper()} ({resp.abc_token_count} score tokens)")
    print(f"Semantic Acoustic LM:  {resp.semantic_token_count} discrete codec frames generated")
    print(f"Flow Matching ODE:     {resp.ode_method_used.upper()} ({resp.ode_steps_used} steps, CFG={resp.cfg_scale_used:.2f})")
    print(f"Boundary Conditioning: {'SYMMETRIC SUB-MS HANN DE-CLICK' if resp.declick_applied else 'BYPASS RAW SAMPLES'}")
    print(f"Signal Dynamics (Peak):{resp.peak_linear:.8f} ({resp.peak_dbfs:.4f} dBFS)")
    print(f"Signal Dynamics (RMS): {resp.rms_dbfs:.4f} dBFS")
    print(f"Acoustic Crest Factor: {resp.crest_factor_db:.4f} dB")
    if resp.abc_notation:
        print("-" * 84)
        print("Generated ABC Notation Excerpt:")
        lines = resp.abc_notation.strip().splitlines()
        preview = lines[:6]
        for l in preview:
            print(f"  {l}")
        if len(lines) > 6:
            print(f"  ... ({len(lines) - 6} additional score lines)")
    print("-" * 84)
    print(f"Effective Style String:\n{resp.effective_style}")
    print("=" * 84 + "\n")


def run_permutation_matrix(engine: Optional[MusicEngine], base_req: GenerationRequest) -> None:
    if engine is None:
        print("\nInitializing Doodle neural engine for permutation sweep...")
        engine = MusicEngine(repo_id=base_req.repo_id, vae_repo_id=base_req.vae_repo_id, device=base_req.device)

    print("\n" + "=" * 84)
    print("            DOODLE ARCHITECTURAL PERMUTATION & ABLATION MATRIX")
    print("=" * 84)

    test_vectors: List[Tuple[str, Dict[str, Any]]] = [
        ("Vocal Standard (Full CoT, Midpoint ODE, CFG=1.0)", {
            "is_instrumental": False, "cot": "full", "ode_method": "midpoint", "cfg_scale": 1.0, "output_path": "perm_vocal_full.wav"
        }),
        ("Vocal Melody-Only (Melody CoT, Heun ODE, CFG=1.0)", {
            "is_instrumental": False, "cot": "melody", "ode_method": "heun", "cfg_scale": 1.0, "output_path": "perm_vocal_melody.wav"
        }),
        ("Vocal Direct Codec (CoT Off, Euler ODE, CFG=1.5)", {
            "is_instrumental": False, "cot": "off", "ode_method": "euler", "cfg_scale": 1.5, "output_path": "perm_vocal_off_cfg.wav"
        }),
        ("Instrumental Arrangement Cues (Full CoT, Midpoint ODE, CFG=1.0)", {
            "is_instrumental": True, "instrumental_branch": "cues", "cot": "full", "ode_method": "midpoint", "cfg_scale": 1.0, "output_path": "perm_inst_cues.wav"
        }),
        ("Instrumental Bare Tags (CoT Off, Midpoint ODE, CFG=1.2)", {
            "is_instrumental": True, "instrumental_branch": "tags_only", "cot": "off", "ode_method": "midpoint", "cfg_scale": 1.2, "output_path": "perm_inst_bare.wav"
        }),
    ]

    results = []
    test_duration = min(base_req.audio_duration, 15.0)

    for idx, (label, overrides) in enumerate(test_vectors):
        print(f"\n[{idx+1}/{len(test_vectors)}] Vector: {label}")
        req = base_req.model_copy(deep=True)
        req.audio_duration = test_duration
        for k, v in overrides.items():
            setattr(req, k, v)

        t0 = time.perf_counter()
        try:
            resp = engine.synthesize(req)
            elapsed = time.perf_counter() - t0
            print(f"  [+] Success | Latency: {elapsed:.2f}s | RTF: {resp.real_time_factor:.2f}x | Peak: {resp.peak_dbfs:.2f} dBFS | RMS: {resp.rms_dbfs:.2f} dBFS")
            results.append((label, True, resp.real_time_factor, resp.peak_vram_gb, resp.crest_factor_db))
        except Exception as e:
            print(f"  [!] Failed: {e}")
            results.append((label, False, 0.0, 0.0, 0.0))

    print("\n" + "=" * 84)
    print("                    PERMUTATION AUDIT CONVERGENCE SUMMARY")
    print("=" * 84)
    for label, ok, rtf, vram, crest in results:
        status_tag = "PASSED" if ok else "FAILED"
        print(f"  [{status_tag}] {label:<60} | RTF: {rtf:5.2f}x | VRAM: {vram:5.2f} GB | Crest: {crest:5.2f} dB")
    print("=" * 84 + "\n")


def display_menu(req: GenerationRequest, engine: Optional[MusicEngine] = None) -> None:
    defaults = get_active_engine_defaults()
    is_inst = req.is_instrumental
    active_lyrics = req.instrumental_lyrics if (is_inst and req.instrumental_branch == "cues") else req.lyrics
    lyrics_status = f"{len(active_lyrics.splitlines())} lines configured" if active_lyrics.strip() else "<Empty Sheet>"
    anchor_tag = "doodle/default.json (Active File)" if has_custom_default_preset() else "Discovered Optimal Baseline (Hardcoded)"
    lead_header = "Acoustic Lead:" if is_inst else "Vocal Profile:"
    lead_content = (req.instrumental_lead or req.vocals) if is_inst else (req.vocal_lead or req.vocals)

    t_sem = req.temperature if req.temperature is not None else defaults["temperature"]
    p_sem = req.top_p if req.top_p is not None else defaults["top_p"]
    k_sem = req.top_k if req.top_k is not None else defaults["top_k"]
    rp_sem = req.repetition_penalty if req.repetition_penalty is not None else defaults["repetition_penalty"]
    cot_mode = (req.cot or defaults["cot"]).upper()
    steps = req.num_inference_steps if req.num_inference_steps is not None else defaults["num_inference_steps"]
    method = (req.ode_method or defaults["ode_method"]).upper()
    cfg = req.cfg_scale if req.cfg_scale is not None else defaults["cfg_scale"]
    core_f = req.vae_core_frames if req.vae_core_frames is not None else defaults["vae_core_frames"]
    halo_f = req.vae_halo_frames if req.vae_halo_frames is not None else defaults["vae_halo_frames"]
    declick_disp = "ENABLED (Symmetric Hann)" if (req.apply_declick if req.apply_declick is not None else defaults["apply_declick"]) else "DISABLED"
    offload_disp = "ENABLED (Sequential Streaming)" if (req.cpu_offload if req.cpu_offload is not None else defaults["cpu_offload"]) else "DISABLED (Resident VRAM)"

    print("\n" + "=" * 84)
    print("                  DOODLE: YUE2 ARCHITECTURAL EXPLORATION HARNESS")
    print(f"                              [{anchor_tag}]")
    print("=" * 84)
    print(" --- PRODUCTION BRIEF (PERSISTENT SONG DRAFT) ---")
    print(f" [M]  Active Modality:       {'INSTRUMENTAL' if is_inst else 'VOCAL SONG'} (Branch: {req.instrumental_branch.upper() if is_inst else 'SONG MASTER'})")
    print(f" [1]  Genre & Subgenre:      {req.genre} / {req.subgenre}")
    print(f" [2]  BPM:                   {req.bpm}")
    print(f" [3]  Key Signature:         {req.key}")
    print(f" [4]  Mood Narrative:        {req.mood}")
    print(f" [5]  {lead_header:<22} {lead_content}")
    print(f" [6]  Arrangement Details:   {req.arrangement}")
    print(f" [7]  Raw Style Override:    {req.raw_prompt if req.raw_prompt else '<Auto-Compiled Tag Vector>'}")
    print(f" [E]  Edit Active Sheet:     {lyrics_status} ({'Instrumental Cues' if (is_inst and req.instrumental_branch == 'cues') else 'Vocal Lyrics'})")
    print(" --- STAGE 1 AUTOREGRESSIVE MoT SAMPLING ---")
    print(f" [8]  Semantic LM Sampling:  T: {t_sem:.2f} | Top-P: {p_sem:.2f} | Top-K: {k_sem} | Rep-Pen: {rp_sem:.3f}")
    print(f" [9]  Symbolic CoT & ABC:    Mode: {cot_mode} (T_abc: {req.abc_temperature:.2f} | K_abc: {req.abc_top_k})")
    print(f" [10] Classifier-Free CFG:   Scale: {cfg:.2f}")
    print(" --- STAGE 2 FLOW MATCHING & STAGE 3 VAE ---")
    print(f" [11] Flow-Match ODE Solver: {method} ({steps} steps)")
    print(f" [12] VAE Receptive Tiling:  Core: {core_f} frames | Halo: {halo_f} frames")
    print(" --- EXECUTION, HARDWARE & DISK ---")
    print(f" [13] Track Length Ceiling:  {req.audio_duration:.2f}s")
    print(f" [14] PRNG Generation Seed:  {req.seed}")
    print(f" [15] Output WAV Path:       {req.output_path}")
    print(f" [16] DSP Boundary De-Click: {declick_disp}")
    print(f" [17] Memory CPU Streaming:  {offload_disp}")
    print("-" * 84)
    print(" --- EXECUTION & INSTRUMENTAL EXPERIMENTATION ---")
    print(f" [G]  Generate Master Track ({'INSTRUMENTAL: ' + req.instrumental_branch.upper() if is_inst else 'VOCAL MASTER'})")
    print(f" [TP] Run Full Permutation Matrix (Ablation Test Suite)")
    print(f" [I]  Configure Instrumental Mode (Branch A: Tags Only | Branch B: Cues)")
    print(f" [T1] Run Bare-Tag Instrumental (Lyrics stripped downwind)")
    print(f" [T2] Run Arrangement-Cue Instrumental (Parenthetical structural directives)")
    print(f" [P]  Preview Compiled [Tags] & Sanitized Lyric Sequence")
    print(f" [T]  Reset Baseline    [L] Load Preset    [S] Save Preset    [D] Direct Save default.json")
    print(f" [Q]  Quit Harness")
    print("=" * 84)


def edit_multiline_sheet(current_text: str, is_inst: bool) -> str:
    print(f"\n--- Edit {'Instrumental Directives' if is_inst else 'Vocal Lyrics'} ---")
    if current_text.strip():
        print(current_text)
    else:
        print("<Currently Empty>")
    print("\nEnter content (Type '__DONE__' on an empty line to finish, or '__CLEAR__' to erase):")
    lines = []
    while True:
        try:
            line = input()
            if line.strip() == "__DONE__":
                break
            if line.strip() == "__CLEAR__":
                return ""
            lines.append(line)
        except EOFError:
            break
    return "\n".join(lines).strip()


def prompt_instrumental_menu(req: GenerationRequest) -> None:
    print("\n" + "-" * 76)
    print("                     INSTRUMENTAL MODE CONFIGURATION")
    print("-" * 76)
    print(f" Current Modality: {'INSTRUMENTAL' if req.is_instrumental else 'VOCAL SONG'}")
    if req.is_instrumental:
        print(f" Active Branch:   {req.instrumental_branch.upper()}")
    print("\n [0] Return to Vocal Song Mode (Singing voice & lyrics enabled)")
    print(" [1] Branch A: Bare-Tag Projection (Song tags preserved, lyrics stripped)")
    print(" [2] Branch B: Arrangement Directives (Parenthetical arrangement directives)")
    print("-" * 76)
    c = input("Select branch [0-2]: ").strip()
    if c == "0":
        req.is_instrumental = False
        req.vocals = req.vocal_lead or req.vocals
        req.output_path = "output_vocal_master.wav"
        print("\nModality set to: VOCAL SONG")
    elif c == "1":
        req.is_instrumental = True
        req.instrumental_branch = "tags_only"
        req.vocals = req.instrumental_lead or ""
        req.output_path = "output_bare_tags.wav"
        print("\nModality set to: INSTRUMENTAL (Branch A: Bare Tags)")
    elif c == "2":
        req.is_instrumental = True
        req.instrumental_branch = "cues"
        req.vocals = req.instrumental_lead or ""
        req.output_path = "output_arrangement_cues.wav"
        if not req.instrumental_lyrics.strip():
            req.instrumental_lyrics = DEFAULT_HARNESS_INSTRUMENTAL_CUES
        print("\nModality set to: INSTRUMENTAL (Branch B: Arrangement Cues)")


def prompt_cot_menu(req: GenerationRequest) -> None:
    curr_cot = (req.cot or "full").upper()
    print("\n" + "-" * 76)
    print("               CHAIN-OF-THOUGHT SYMBOLIC PLANNING REGIME")
    print("-" * 76)
    print(f" Current Mode: {curr_cot}")
    print("\n [1] FULL   - Generate chord-annotated ABC score prior to acoustic synthesis")
    print(" [2] MELODY - Generate melody-only ABC score (optimal for covers/strict leads)")
    print(" [3] OFF    - Direct synthesis without symbolic planning")
    print(" [4] Tune ABC Sampling Hyperparameters (Temperature, Top-P, Top-K, Rep-Pen)")
    print("-" * 76)
    sel = input("Select operation [1-4]: ").strip()
    if sel == "1":
        req.cot = "full"
    elif sel == "2":
        req.cot = "melody"
    elif sel == "3":
        req.cot = "off"
    elif sel == "4":
        t = input(f"Enter ABC Temperature [{req.abc_temperature}]: ").strip()
        if t:
            req.abc_temperature = float(t)
        p = input(f"Enter ABC Top-P [{req.abc_top_p}]: ").strip()
        if p:
            req.abc_top_p = float(p)
        k = input(f"Enter ABC Top-K [{req.abc_top_k}]: ").strip()
        if k and k.isdigit():
            req.abc_top_k = int(k)
        rp = input(f"Enter ABC Repetition Penalty [{req.abc_repetition_penalty}]: ").strip()
        if rp:
            req.abc_repetition_penalty = float(rp)


def prompt_ode_method(req: GenerationRequest) -> None:
    curr_ode = (req.ode_method or "midpoint").upper()
    print("\n" + "-" * 76)
    print("                FLOW MATCHING NUMERICAL INTEGRATION METHOD")
    print("-" * 76)
    print(f" Current Method: {curr_ode}")
    print("\n [1] MIDPOINT - Second-order Runge-Kutta midpoint integrator (Reference standard)")
    print(" [2] HEUN     - Explicit trapezoidal predictor-corrector")
    print(" [3] EULER    - First-order tangent field integrator")
    print("-" * 76)
    sel = input("Select solver [1-3]: ").strip()
    if sel == "1":
        req.ode_method = "midpoint"
    elif sel == "2":
        req.ode_method = "heun"
    elif sel == "3":
        req.ode_method = "euler"


def run_interactive_harness(engine: Optional[MusicEngine], initial_req: Optional[GenerationRequest] = None) -> None:
    req = initial_req if initial_req is not None else create_default_harness_request()
    while True:
        display_menu(req, engine)
        choice = input("Select action or field: ").strip().upper()
        if choice == "M":
            req.is_instrumental = not req.is_instrumental
            if req.is_instrumental:
                req.vocals = req.instrumental_lead or ""
                req.output_path = "output_instrumental.wav"
            else:
                req.vocals = req.vocal_lead or req.vocals
                req.output_path = "output_vocal_master.wav"
            print(f"\nSwitched modality to: {'INSTRUMENTAL' if req.is_instrumental else 'VOCAL SONG'}")
        elif choice == "I":
            prompt_instrumental_menu(req)
        elif choice == "1":
            g = input(f"Enter Genre [{req.genre}]: ").strip()
            if g:
                req.genre = g
            sg = input(f"Enter Subgenre [{req.subgenre}]: ").strip()
            if sg:
                req.subgenre = sg
        elif choice == "2":
            b = input(f"Enter BPM (30 - 300, 0 for unmetered) [{req.bpm}]: ").strip()
            if b.isdigit() and (int(b) == 0 or 30 <= int(b) <= 300):
                req.bpm = int(b)
        elif choice == "3":
            k = input(f"Enter Key Signature [{req.key}]: ").strip()
            if k:
                req.key = k
        elif choice == "4":
            m = input(f"Enter Mood Narrative [{req.mood}]: ").strip()
            if m:
                req.mood = m
        elif choice == "5":
            if req.is_instrumental:
                curr = req.instrumental_lead or req.vocals
                v = input(f"Enter Instrumental Lead / Acoustic Character [{curr}]: ").strip()
                if v:
                    req.instrumental_lead = v
                    req.vocals = v
            else:
                curr = req.vocal_lead or req.vocals
                v = input(f"Enter Vocal Profile & Character [{curr}]: ").strip()
                if v:
                    req.vocal_lead = v
                    req.vocals = v
        elif choice == "6":
            a = input(f"Enter Arrangement Details [{req.arrangement}]: ").strip()
            if a:
                req.arrangement = a
        elif choice == "7":
            r = input("Enter Raw Style override (empty to reset to auto-compiled tags): ").strip()
            req.raw_prompt = r if r else None
        elif choice == "E":
            if req.is_instrumental and req.instrumental_branch == "cues":
                req.instrumental_lyrics = edit_multiline_sheet(req.instrumental_lyrics, is_inst=True)
            else:
                req.lyrics = edit_multiline_sheet(req.lyrics, is_inst=False)
        elif choice == "8":
            t = input(f"Enter Semantic Temperature [{req.temperature}]: ").strip()
            if t:
                req.temperature = float(t)
            p = input(f"Enter Semantic Top-P [{req.top_p}]: ").strip()
            if p:
                req.top_p = float(p)
            k = input(f"Enter Semantic Top-K [{req.top_k}]: ").strip()
            if k and k.isdigit():
                req.top_k = int(k)
            rp = input(f"Enter Repetition Penalty [{req.repetition_penalty}]: ").strip()
            if rp:
                req.repetition_penalty = float(rp)
        elif choice == "9":
            prompt_cot_menu(req)
        elif choice == "10":
            c = input(f"Enter CFG Scale [{req.cfg_scale}]: ").strip()
            if c:
                req.cfg_scale = float(c)
        elif choice == "11":
            prompt_ode_method(req)
            st = input(f"Enter Number of ODE Steps [{req.num_inference_steps}]: ").strip()
            if st and st.isdigit():
                req.num_inference_steps = int(st)
        elif choice == "12":
            cf = input(f"Enter VAE Core Frames (128-4096) [{req.vae_core_frames}]: ").strip()
            if cf and cf.isdigit():
                req.vae_core_frames = int(cf)
            hf = input(f"Enter VAE Halo Frames (>=16) [{req.vae_halo_frames}]: ").strip()
            if hf and hf.isdigit():
                req.vae_halo_frames = int(hf)
        elif choice in ("13", "DUR"):
            d = input(f"Enter Duration Ceiling (seconds) [{req.audio_duration:.2f}]: ").strip()
            if d:
                req.audio_duration = float(d)
        elif choice in ("14", "18"):
            sd = input(f"Enter PRNG Seed [{req.seed}]: ").strip()
            if sd and sd.isdigit():
                req.seed = int(sd)
        elif choice in ("15", "19"):
            dst = input(f"Enter Output WAV Path [{req.output_path}]: ").strip()
            if dst:
                req.output_path = dst
        elif choice in ("16", "21"):
            req.apply_declick = not req.apply_declick
        elif choice in ("17", "22"):
            req.cpu_offload = not req.cpu_offload
        elif choice == "P":
            branch_label = req.instrumental_branch.upper() if req.is_instrumental else "VOCAL SONG"
            print(f"\n--- Compiled Style Tags ---\n{req.compile_style()}\n")
            print(f"--- Sanitized Sequence (Branch: {branch_label}) ---\n{req.sanitize_lyrics()}\n")
            print(f"--- Full Ingest Prompt ---\n{req.compile_full_text()}\n")
            input("Press Enter to continue...")
        elif choice == "TP":
            run_permutation_matrix(engine, req)
        elif choice == "T1":
            if engine is None:
                print("\nInitializing Doodle neural engine...")
                engine = MusicEngine(repo_id=req.repo_id, vae_repo_id=req.vae_repo_id, device=req.device)
            test_req = req.model_copy(deep=True)
            test_req.is_instrumental = True
            test_req.instrumental_branch = "tags_only"
            test_req.vocals = test_req.instrumental_lead or ""
            test_req.output_path = "output_bare_tags.wav"
            print("\nExecuting Branch A: Bare-Tag Projection...")
            try:
                resp = engine.synthesize(test_req)
                print_telemetry(resp, test_req)
            except Exception as e:
                print(f"Branch A execution failed: {e}", file=sys.stderr)
                traceback.print_exc()
        elif choice == "T2":
            if engine is None:
                print("\nInitializing Doodle neural engine...")
                engine = MusicEngine(repo_id=req.repo_id, vae_repo_id=req.vae_repo_id, device=req.device)
            test_req = req.model_copy(deep=True)
            test_req.is_instrumental = True
            test_req.instrumental_branch = "cues"
            test_req.vocals = test_req.instrumental_lead or ""
            test_req.output_path = "output_arrangement_cues.wav"
            if not test_req.instrumental_lyrics.strip():
                test_req.instrumental_lyrics = DEFAULT_HARNESS_INSTRUMENTAL_CUES
            print("\nExecuting Branch B: Arrangement Directives...")
            try:
                resp = engine.synthesize(test_req)
                print_telemetry(resp, test_req)
            except Exception as e:
                print(f"Branch B execution failed: {e}", file=sys.stderr)
                traceback.print_exc()
        elif choice == "T":
            default_fixture = create_default_harness_request()
            req = default_fixture
            print("\nReset active configuration to discovered baseline.")
        elif choice == "L":
            p_path = input("Enter JSON preset to load: ").strip()
            if not p_path:
                continue
            chosen_path = Path(p_path)
            if not chosen_path.is_absolute() and not chosen_path.exists():
                candidate_preset = ROOT_DIR / "presets" / p_path
                if candidate_preset.exists():
                    chosen_path = candidate_preset
                else:
                    candidate_root = ROOT_DIR / p_path
                    if candidate_root.exists():
                        chosen_path = candidate_root
            try:
                req = GenerationRequest.load_preset(chosen_path)
                print(f"Preset loaded successfully from {chosen_path}")
            except Exception as e:
                print(f"Preset load failure: {e}")
        elif choice == "S":
            p_path = input(f"Enter destination JSON preset path (e.g., {DEFAULT_PRESET_FILENAME}): ").strip()
            if not p_path:
                continue
            if not p_path.endswith(".json"):
                p_path = f"{p_path}.json"
            target = Path(p_path)
            if not target.is_absolute():
                if target.name == DEFAULT_PRESET_FILENAME:
                    target = ROOT_DIR / DEFAULT_PRESET_FILENAME
                else:
                    target = ROOT_DIR / "presets" / target.name
            try:
                req.save_preset(target)
                print(f"Preset saved successfully to {target}")
            except Exception as e:
                print(f"Preset save failure: {e}")
        elif choice == "D":
            target = ROOT_DIR / DEFAULT_PRESET_FILENAME
            try:
                req.save_preset(target)
                print(f"Authoritative default.json updated at {target}")
            except Exception as e:
                print(f"Preset save failure: {e}")
        elif choice == "G":
            if engine is None:
                print("\nInitializing Doodle neural engine...")
                engine = MusicEngine(repo_id=req.repo_id, vae_repo_id=req.vae_repo_id, device=req.device)
            cot_tag = (req.cot or "full").upper()
            mode_tag = f"INSTRUMENTAL ({req.instrumental_branch.upper()})" if req.is_instrumental else "VOCAL MASTER"
            print(
                f"\nSynthesizing ({mode_tag}, Ceiling={req.audio_duration:.2f}s, Steps={req.num_inference_steps}, CoT={cot_tag})..."
            )
            try:
                resp = engine.synthesize(req)
                print_telemetry(resp, req)
            except Exception as e:
                print(f"Synthesis failed: {e}", file=sys.stderr)
                traceback.print_exc()
        elif choice == "Q":
            sys.exit(0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Modality Exploration & Ablation Harness for Doodle (YuE2).")
    parser.add_argument("--batch", action="store_true", help="Run non-interactive generation pass.")
    parser.add_argument("--test_permutations", action="store_true", help="Run automated permutation ablation suite.")
    parser.add_argument("--blank", action="store_true", help="Start with blank fields rather than baseline fixture.")
    parser.add_argument("--instrumental", action="store_true", help="Engage instrumental mode.")
    parser.add_argument("--branch", type=str, choices=["tags_only", "cues"], default="cues")
    parser.add_argument("--genre", type=str, default=None)
    parser.add_argument("--bpm", type=int, default=None)
    parser.add_argument("--key", type=str, default=None)
    parser.add_argument("--mood", type=str, default=None)
    parser.add_argument("--vocals", type=str, default=None)
    parser.add_argument("--vocal_lead", type=str, default=None)
    parser.add_argument("--inst_lead", dest="instrumental_lead", type=str, default=None)
    parser.add_argument("--arrangement", type=str, default=None)
    parser.add_argument("--raw_prompt", type=str, default=None)
    parser.add_argument("--lyrics", type=str, default=None)
    parser.add_argument("--inst_lyrics", type=str, default=None)
    parser.add_argument("--cot", type=str, choices=SUPPORTED_COT, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--steps", dest="num_inference_steps", type=int, default=None)
    parser.add_argument("--solver", dest="ode_method", type=str, choices=SUPPORTED_ODE_METHODS, default=None)
    parser.add_argument("--cfg", dest="cfg_scale", type=float, default=None)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--no_declick", action="store_true", default=False)
    parser.add_argument("--cpu_offload", action="store_true", default=None)
    parser.add_argument("--load_preset", type=str, default=None)
    parser.add_argument("--save_preset", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--repo_id", type=str, default=None)
    parser.add_argument("--vae_repo_id", type=str, default=None)
    args = parser.parse_args()

    if args.load_preset:
        req = GenerationRequest.load_preset(Path(args.load_preset))
    elif args.blank:
        req = GenerationRequest()
    else:
        req = create_default_harness_request()

    if args.instrumental:
        req.is_instrumental = True
    if args.branch is not None:
        req.instrumental_branch = args.branch
    if args.genre is not None:
        req.genre = args.genre
    if args.bpm is not None:
        req.bpm = args.bpm
    if args.key is not None:
        req.key = args.key
    if args.mood is not None:
        req.mood = args.mood
    if args.vocals is not None:
        req.vocals = args.vocals
    if args.vocal_lead is not None:
        req.vocal_lead = args.vocal_lead
    if args.instrumental_lead is not None:
        req.instrumental_lead = args.instrumental_lead
    if args.arrangement is not None:
        req.arrangement = args.arrangement
    if args.raw_prompt is not None:
        req.raw_prompt = args.raw_prompt
    if args.cot is not None:
        req.cot = args.cot
    if args.temperature is not None:
        req.temperature = args.temperature
    if args.top_p is not None:
        req.top_p = args.top_p
    if args.top_k is not None:
        req.top_k = args.top_k
    if args.num_inference_steps is not None:
        req.num_inference_steps = args.num_inference_steps
    if args.ode_method is not None:
        req.ode_method = args.ode_method
    if args.cfg_scale is not None:
        req.cfg_scale = args.cfg_scale
    if args.duration is not None:
        req.audio_duration = args.duration
    if args.seed is not None:
        req.seed = args.seed
    if args.output is not None:
        req.output_path = args.output
    if args.no_declick:
        req.apply_declick = False
    if args.cpu_offload is not None:
        req.cpu_offload = args.cpu_offload
    if args.device is not None:
        req.device = args.device
    if args.repo_id is not None:
        req.repo_id = args.repo_id
    if args.vae_repo_id is not None:
        req.vae_repo_id = args.vae_repo_id
    if args.lyrics is not None:
        p = Path(args.lyrics)
        req.lyrics = p.read_text(encoding="utf-8") if p.is_file() else args.lyrics
    if args.inst_lyrics is not None:
        p = Path(args.inst_lyrics)
        req.instrumental_lyrics = p.read_text(encoding="utf-8") if p.is_file() else args.inst_lyrics

    if args.save_preset:
        target = Path(args.save_preset)
        if not target.is_absolute() and target.name == DEFAULT_PRESET_FILENAME:
            target = ROOT_DIR / DEFAULT_PRESET_FILENAME
        req.save_preset(target)
        print(f"Preset exported to {target}")
        sys.exit(0)

    if args.test_permutations:
        engine = MusicEngine(repo_id=req.repo_id, vae_repo_id=req.vae_repo_id, device=req.device)
        run_permutation_matrix(engine, req)
    elif args.batch:
        engine = MusicEngine(repo_id=req.repo_id, vae_repo_id=req.vae_repo_id, device=req.device)
        resp = engine.synthesize(req)
        print_telemetry(resp, req)
    else:
        run_interactive_harness(engine=None, initial_req=req)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception:
        traceback.print_exc()
        sys.exit(1)