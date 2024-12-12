import hashlib
import hmac

import zmq
import zmq.asyncio


def sign(key: bytes, msg: bytes) -> bytes:
    """Compute the HMAC digest of msg, given signing key"""

    return hmac.HMAC(
        key,
        msg,
        digestmod=hashlib.sha256,
    ).digest()


def check_signed(key: bytes, sig: bytes, msg: bytes) -> bool:
    """Check if signature (HMAC digest) matches."""

    correct_sig = sign(key, msg)
    return hmac.compare_digest(sig, correct_sig)


def send_signed(socket: zmq.Socket, key: bytes, msg: bytes):
    """Send signed message to socket."""

    sig = sign(key, msg)
    socket.send_multipart((sig, msg), copy=False)


def recv_signed(socket: zmq.Socket, key: bytes) -> bytes:
    """Get signed message from socket."""

    sig, msg = socket.recv_multipart(copy=False)
    if not check_signed(key, sig, msg.buffer):
        raise ValueError("Message signature is invalid.")
    return msg.buffer


async def send_signed_async(socket: zmq.asyncio.Socket, key: bytes,
                            msg: bytes):
    """Send signed message to asyncio socket."""

    sig = sign(key, msg)
    await socket.send_multipart((sig, msg), copy=False)


async def recv_signed_async(socket: zmq.asyncio.Socket, key: bytes) -> bytes:
    """Get signed message from asyncio socket."""

    sig, msg = await socket.recv_multipart(copy=False)
    if not check_signed(key, sig, msg.buffer):
        raise ValueError("Message signature is invalid.")
    return msg.buffer
