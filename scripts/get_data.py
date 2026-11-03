"""Descarga Time-MMD (solo dominio Energy) y verifica el hash. Time-MMD no declara licencia de
redistribucion en el repositorio, por eso los datos no se versionan aqui."""
import subprocess, shutil, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tsbase import config as C
from tsbase.data import sha256

COMMIT = "00281e2d86058286d5548b15a7670e8eda57ef62"
HASHES = {
    "numerical/Energy/Energy.csv": "94313cefb3459f04b58ec814da5d817a694e767cc52de0ce09d4de4efdad02ee",
    "textual/Energy/Energy_report.csv": "36fa229dca003f40d4fb8c816a8ddfde2e020fca1f14572b3ce81a368e948fe8",
    "textual/Energy/Energy_search.csv": "f9023e6fcb635731a74f248dcf5ceb8aeca48870f3fc559f2173b7aae790b595"
}

def main():
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["git", "clone", "https://github.com/AdityaLab/Time-MMD.git", tmp + "/t"], check=True)
        subprocess.run(["git", "-C", tmp + "/t", "checkout", COMMIT], check=True)
        for kind, name in (("numerical", "Energy"), ("textual", "Energy")):
            dst = C.DATA_DIR / kind / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(Path(tmp) / "t" / kind / name, dst, dirs_exist_ok=True)
    for rel, h in HASHES.items():
        got = sha256(C.DATA_DIR / rel)
        print(("OK  " if got == h else "FALLA"), rel, got)
        assert got == h, f"hash distinto en {rel}"

if __name__ == "__main__":
    main()
