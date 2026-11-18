"""Run every program one after another (close each plot window to continue)."""
import glob, runpy, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
for f in sorted(glob.glob("p[0-9][0-9]_*.py")):
    print("\n" + "=" * 80 + f"\nRUNNING {f}\n" + "=" * 80)
    runpy.run_path(f, run_name="__main__")
