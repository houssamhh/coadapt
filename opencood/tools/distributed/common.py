import pickle
import socket
import struct
import time
from typing import Dict, List, Optional, Tuple

import numpy as np


HEADER_STRUCT = struct.Struct("!I")


def send_message(sock: socket.socket, payload: dict) -> None:
    body = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(HEADER_STRUCT.pack(len(body)))
    sock.sendall(body)


def recv_message(sock: socket.socket) -> dict:
    header = _recvall(sock, HEADER_STRUCT.size)
    (size,) = HEADER_STRUCT.unpack(header)
    body = _recvall(sock, size)
    return pickle.loads(body)


def _recvall(sock: socket.socket, size: int) -> bytes:
    chunks = []
    read = 0
    while read < size:
        data = sock.recv(size - read)
        if not data:
            raise ConnectionError("socket closed while reading frame")
        chunks.append(data)
        read += len(data)
    return b"".join(chunks)


def connect_with_retry(host: str, port: int, timeout_s: float = 60.0) -> socket.socket:
    start = time.time()
    last_error = None
    while time.time() - start < timeout_s:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.connect((host, port))
            return sock
        except OSError as exc:
            last_error = exc
            sock.close()
            time.sleep(0.5)
    raise TimeoutError(f"could not connect to {host}:{port} ({last_error})")


def make_server_socket(bind_host: str, port: int, backlog: int = 64) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((bind_host, port))
    sock.listen(backlog)
    return sock


def ordered_agent_ids(base_data_dict: Dict[str, dict]) -> List[str]:
    ego_id = None
    others = []
    for cav_id, cav_content in base_data_dict.items():
        if bool(cav_content.get("ego", False)):
            ego_id = cav_id
        else:
            others.append(cav_id)
    others = sorted(others, key=str)
    if ego_id is None:
        return others
    return [ego_id] + others


def select_cav_for_slot(base_data_dict: Dict[str, dict], agent_slot: int) -> Tuple[Optional[str], Optional[dict]]:
    ordered = ordered_agent_ids(base_data_dict)
    if agent_slot < 0 or agent_slot >= len(ordered):
        return None, None
    cav_id = ordered[agent_slot]
    return cav_id, base_data_dict[cav_id]


def np_nbytes(x: Optional[np.ndarray]) -> int:
    if x is None:
        return 0
    return int(x.nbytes)
