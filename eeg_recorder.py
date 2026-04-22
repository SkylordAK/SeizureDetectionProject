"""
eeg_recorder.py
---------------
Records raw EEG from the Muse S (Athena) device to labelled CSV files.
Uses the same OSC/pythonosc connection pattern as the Motor Imagery project.

USAGE:
    1. Start Mind Monitor (or MuseLSL) on your phone and point OSC stream
       to this machine's IP at port 5239.
    2. Run:  python eeg_recorder.py
    3. Send Marker 1 (from Mind Monitor or a UDP sender) to START recording.
    4. Send Marker 2 to STOP recording and exit.

OUTPUT:
    Recordings/<SESSION_NAME>.<timestamp>.csv
    Columns: timestamp, RAW_TP9, RAW_AF7, RAW_AF8, RAW_TP10
"""

import os
from datetime import datetime
from timeit import default_timer as timer
from pythonosc import dispatcher, osc_server

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
IP   = '0.0.0.0'    # Must match your Mind Monitor OSC target IP
PORT = 5239
OUTPUT_DIR = 'Recordings'

# Recording session definition:
# key  = label written in filename (Normal, Seizure, Epilepsy, break)
# value = duration in seconds for that segment
rec_dict = {
    'break0'   : 10,    # Initial rest / warm-up
    'Normal'   : 60,    # Resting, eyes open — baseline EEG
    'break1'   : 10,
    'Epilepsy' : 60,    # E.g. hyperventilation-induced or known epileptic period
    'break2'   : 10,
    'Seizure'  : 60,    # Active seizure period (clinician/caregiver triggered)
    'break3'   : 10,
}

CSV_HEADER = 'timestamp,RAW_TP9,RAW_AF7,RAW_AF8,RAW_TP10\n'
# ──────────────────────────────────────────────────────────────────────────────

recording     = False
initial_read  = True
current_event = 0
lock          = False
row           = 0
start         = timer()
f             = None

os.makedirs(OUTPUT_DIR, exist_ok=True)


def open_new_file(label: str) -> object:
    ts   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join(OUTPUT_DIR, f"{label}.{ts}.csv")
    fh   = open(path, 'w', newline='')
    fh.write(CSV_HEADER)
    print(f"  → Recording segment: [{label}]  ({rec_dict.get(label, '?')}s)  → {path}")
    return fh


def eeg_handler(address: str, *args):
    global recording, initial_read, current_event, lock, row, start, f

    if not recording:
        return

    # First sample after START: open warm-up file
    if initial_read:
        initial_read = False
        start = timer()
        ev = list(rec_dict.items())[current_event][0]
        f  = open_new_file(ev)
        row = 0
        return

    elapsed = timer() - start
    seg_dur = list(rec_dict.items())[current_event][1]

    if elapsed >= seg_dur and not lock:
        lock = True
        # Close the current segment file
        if f:
            f.close()
        current_event += 1

        if current_event >= len(rec_dict):
            print("\n✅  All segments recorded. Wrapping up...")
            recording = False
            lock = False
            return

        ev   = list(rec_dict.items())[current_event][0]
        secs = list(rec_dict.items())[current_event][1]
        f    = open_new_file(ev)
        row  = 0
        start = timer()
        print(f"     Next segment in {secs}s")
        lock = False

    else:
        if not lock and f:
            ts_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            line   = f"{ts_str}"
            for i in range(4):
                line += f",{args[i]:.4f}"
            line += "\n"
            f.write(line)
            row += 1


def marker_handler(address: str, i):
    global recording, initial_read, current_event, start

    marker = address[-1]

    if marker == '1':
        recording    = True
        initial_read = True
        current_event = 0
        start = timer()
        print("🔴  Recording STARTED — waiting for first EEG sample...")

    elif marker == '2':
        recording = False
        if f:
            f.close()
        print("⏹  Recording STOPPED.")
        server.shutdown()


if __name__ == '__main__':
    disp = dispatcher.Dispatcher()
    disp.map('/muse/eeg', eeg_handler)
    disp.map('/eeg', eeg_handler)
    disp.map('/Marker/*', marker_handler)

    server = osc_server.ThreadingOSCUDPServer((IP, PORT), disp)
    print("=" * 60)
    print("  Muse S EEG Recorder — Epilepsy Study")
    print(f"  Listening on UDP  {IP}:{PORT}")
    print("  Send Marker 1 → Start recording")
    print("  Send Marker 2 → Stop  recording")
    print("=" * 60)
    server.serve_forever()
