"""Passive tap on the egress path plus a shaped wide area link.

Every client session is routed through this relay. The relay never decrypts
anything. It parses the five byte TLS record header off the ciphertext stream,
timestamps each record at the observation point, and forwards it after a delay
that reproduces propagation, jitter and serialisation on a shaped link. The
recorded tuples are therefore measurements of an emulated link rather than
values invented by the author.
"""
from __future__ import annotations

import random
import socket
import threading
import time
from dataclasses import dataclass, field

from config import (LINK_DOWNLINK_MBPS, LINK_ONEWAY_MS_JITTER, LINK_ONEWAY_MS_MEAN,
                    LINK_UPLINK_MBPS)

UP = 0    # client to endpoint
DOWN = 1  # endpoint to client


@dataclass
class Observation:
    """Per connection list of (wire_time, direction, tls_record_bytes)."""
    records: list[tuple[float, int, int]] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, when: float, direction: int, size: int) -> None:
        with self.lock:
            self.records.append((when, direction, size))


class _Shaper:
    """Serialises a direction of the link and adds propagation with jitter."""

    def __init__(self, mbps: float, rng: random.Random):
        self.bits_per_second = mbps * 1e6
        self.free_at = time.perf_counter()
        self.rng = rng

    def release(self, nbytes: int) -> None:
        now = time.perf_counter()
        start = max(now, self.free_at)
        self.free_at = start + (nbytes * 8) / self.bits_per_second
        oneway = self.rng.gauss(LINK_ONEWAY_MS_MEAN, LINK_ONEWAY_MS_JITTER) / 1000.0
        oneway = max(0.0015, oneway)
        deliver_at = self.free_at + oneway
        gap = deliver_at - time.perf_counter()
        if gap > 0:
            time.sleep(gap)


def _pump(src: socket.socket, dst: socket.socket, direction: int,
          obs: Observation, shaper: _Shaper, t0: float) -> None:
    buffer = bytearray()
    try:
        while True:
            block = src.recv(65536)
            if not block:
                break
            buffer += block
            while len(buffer) >= 5:
                length = (buffer[3] << 8) | buffer[4]
                total = 5 + length
                if len(buffer) < total:
                    break
                record = bytes(buffer[:total])
                del buffer[:total]
                obs.add(time.perf_counter() - t0, direction, total)
                shaper.release(total)
                dst.sendall(record)
    except OSError:
        pass
    finally:
        try:
            dst.shutdown(socket.SHUT_WR)
        except OSError:
            pass


class EgressTap:
    """Listens locally, forwards to the inference endpoint, records the wire."""

    def __init__(self, upstream_port: int, host: str = "127.0.0.1", seed: int = 0):
        self.upstream = (host, upstream_port)
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listener.bind((host, 0))
        self.listener.listen(256)
        self.port = self.listener.getsockname()[1]
        self._pending: dict[int, Observation] = {}
        self._lock = threading.Lock()
        self._seed = seed
        self._running = True
        threading.Thread(target=self._accept_loop, daemon=True).start()

    def register(self, token: int) -> Observation:
        obs = Observation()
        with self._lock:
            self._pending[token] = obs
        return obs

    def _accept_loop(self) -> None:
        while self._running:
            try:
                client, _ = self.listener.accept()
            except OSError:
                return
            threading.Thread(target=self._serve, args=(client,), daemon=True).start()

    def _serve(self, client: socket.socket) -> None:
        try:
            header = b""
            while len(header) < 8:
                block = client.recv(8 - len(header))
                if not block:
                    client.close()
                    return
                header += block
            token = int.from_bytes(header, "big")
            with self._lock:
                obs = self._pending.pop(token, None)
            if obs is None:
                obs = Observation()
            upstream = socket.create_connection(self.upstream)
            client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            upstream.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            rng = random.Random(self._seed ^ token)
            t0 = time.perf_counter()
            up_shaper = _Shaper(LINK_UPLINK_MBPS, rng)
            down_shaper = _Shaper(LINK_DOWNLINK_MBPS, rng)
            a = threading.Thread(target=_pump, args=(client, upstream, UP, obs, up_shaper, t0))
            b = threading.Thread(target=_pump, args=(upstream, client, DOWN, obs, down_shaper, t0))
            a.start(); b.start(); a.join(); b.join()
            client.close(); upstream.close()
        except OSError:
            try:
                client.close()
            except OSError:
                pass

    def close(self) -> None:
        self._running = False
        try:
            self.listener.close()
        except OSError:
            pass
