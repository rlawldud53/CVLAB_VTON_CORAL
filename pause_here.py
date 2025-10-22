from pathlib import Path
import time

def pause(acc):
    if Path(".pause_all").exists():
        if acc.local_process_index == 0:
            import pdb; pdb.set_trace()
        else:
            while Path(".pause_all").exists():
                time.sleep(0.2)
