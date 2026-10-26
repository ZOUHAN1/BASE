import hashlib
import json
import ctypes
import csv
import re
import subprocess
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def dump_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def cuda_uuid(logical_index):
    runtime = ctypes.CDLL('libcudart.so.12')
    runtime.cudaDeviceGetPCIBusId.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    runtime.cudaDeviceGetPCIBusId.restype = ctypes.c_int
    value = ctypes.create_string_buffer(64)
    code = runtime.cudaDeviceGetPCIBusId(value, len(value), logical_index)
    assert code == 0, ('cudaDeviceGetPCIBusId', code)

    def pci_key(address):
        match = re.fullmatch(r'([0-9a-f]+):([0-9a-f]+):([0-9a-f]+)\.([0-9a-f]+)', address.strip().lower())
        assert match, address
        return tuple(int(part, 16) for part in match.groups())

    listing = subprocess.check_output([
        'nvidia-smi', '--query-gpu=uuid,pci.bus_id', '--format=csv,noheader,nounits',
    ], text=True)
    devices = {pci_key(bus): gpu.strip() for gpu, bus in csv.reader(listing.splitlines())}
    return devices[pci_key(value.value.decode('ascii'))]
