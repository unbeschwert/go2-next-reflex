import socket
import struct
import warnings
from math import prod
from typing import Sequence

import cbor2
import numpy as np
import numpy.typing as npt


class FPGAClient:
    """Encapsulates logic to talk to the SPLEAT driver on an FPGA."""

    _fpga_host: str
    _fpga_port: int
    _last_msg_id: int | None
    _socket: socket.socket | None
    _shape_in: Sequence[int]
    _shape_out: Sequence[int]

    def __init__(
        self, host: str, port: int, shape_in: Sequence[int], shape_out: Sequence[int]
    ):
        self._fpga_host = host
        self._fpga_port = port
        self._last_msg_id = None
        self._socket = None
        self._shape_in = shape_in
        self._shape_out = shape_out

    def connect(self):
        if self._socket is not None:
            warnings.warn("Socket already connected. Doing nothing.")
            return

        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._socket.connect((self._fpga_host, self._fpga_port))
        return

    def close(self):
        if self._socket is None:
            warnings.warn("Nothing to close. Doing nothing.")
            return

        self._socket.close()
        self._socket = None
        return

    def send_tensor(self, t: npt.ArrayLike):
        """Sends a tensor and returns the duration it tooks to send it."""
        arr = np.asarray(t, dtype=np.float32)

        if self._socket is None:
            raise RuntimeError("FPGAClient is not connected. Unable to send data.")

        if tuple(arr.shape) != tuple(self._shape_in):
            raise RuntimeError("Unable to send tensor: wrong shape.")

        msg_id = 0
        if self._last_msg_id is not None:
            msg_id = self._last_msg_id + 1

        msg = cbor2.dumps({"id": msg_id, "payload": arr.tobytes()})

        # Send message length as uint32_t big endian.
        self._socket.sendall(struct.pack("!I", len(msg)))
        self._socket.sendall(msg)

        self._last_msg_id = msg_id

    def receive_tensor(self):
        """Receives a tensor and returns it with duration it took to receive it."""

        if self._socket is None:
            raise RuntimeError("FPGAClient is not connected. Unable to receive data.")

        (response_length,) = struct.unpack("!I", _recv_exactly(self._socket, 4))
        response = cbor2.loads(_recv_exactly(self._socket, response_length))

        if "error" in response and response["error"]:
            raise RuntimeError(f"Error from server: {response['error']}")

        if "payload" not in response:
            raise RuntimeError(f"Invalid response from server: {response}")

        t = np.frombuffer(response["payload"], dtype=np.float32)
        if t.shape[0] != prod(self._shape_out):
            raise RuntimeError("Received a tensor with the wrong size from the server.")
        t = t.reshape(self._shape_out)

        return t, response["latency"]


def _recv_exactly(sock: socket.socket, n: int):
    """Helper for receiving a fixed amount of data through a TCP socket."""

    buffer = bytearray(n)
    view = memoryview(buffer)  # necessary to perform 0-copy slicing
    received = 0
    while received < n:
        count = sock.recv_into(view[received:])
        if count == 0:  # connection closed
            raise ConnectionError(
                f"socket closed while expecting {n - received} more bytes"
            )
        received += count
    return buffer
