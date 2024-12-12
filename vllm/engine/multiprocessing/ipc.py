import hashlib
import hmac
import secrets

import zmq
import zmq.asyncio

# TODO: switch to SECRET_KEY = secrets.token_bytes(16) 
# and pass the SECRET_KEY to the background process.
SECRET_KEY = b"my_key"

def sign(msg: bytes) -> bytes:
    """Compute the HMAC digest of msg, given signing key `key`"""
    return hmac.HMAC(
        SECRET_KEY,
        msg,
        digestmod=hashlib.sha256,
    ).digest()

def check_signed(sig: bytes, msg: bytes) -> bool:
    correct_sig = sign(msg)
    return hmac.compare_digest(sig, correct_sig)

def send_signed(socket: zmq.Socket, msg: bytes):
    """Send signed message to socket."""

    sig = sign(msg)
    socket.send_multipart((sig, msg), copy=False)

def recv_signed(socket: zmq.Socket):
    """Get signed message from socket."""

    sig, msg = socket.recv_multipart(copy=False)
    if not check_signed(sig, msg):
        raise ValueError("Message signature is invalid.")
    return msg

async def send_signed_async(socket: zmq.asyncio.Socket, msg: bytes):
    """Send signed message to asyncio socket."""

    sig = sign(msg)
    await socket.send_multipart((sig, msg), copy=False)

async def recv_signed_async(socket: zmq.asyncio.Socket):
    """Get signed message from asyncio socket."""

    sig, msg = await socket.recv_multipart(copy=False)
    if not check_signed(sig, msg):
        raise ValueError("Message signature is invalid.")
    return msg