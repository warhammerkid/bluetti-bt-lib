from dataclasses import dataclass
from enum import IntEnum, unique
import hashlib
import os
import struct
import sys
from typing import Optional, cast

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import (
    decode_dss_signature,
    encode_dss_signature,
)
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.padding import PKCS7


def aes_decrypt(data: bytes, aes_key: bytes, iv: Optional[bytes] = None) -> bytes:
    """Decrypt data using non-standard AES-CBC.

    Format: [length:2][iv_seed:4?][encrypted_data:N]

    Args:
        data: Encrypted message
        aes_key: 16-byte AES key
        iv: Optional 16-byte IV. If None, IV seed will be extracted from data

    Returns:
        Decrypted plaintext (without padding)
    """

    view = memoryview(data)
    data_len = struct.unpack("!H", view[:2])[0]

    # If no IV is given, derive it from the 4 bytes after the length header
    if iv is None:
        iv_seed = view[2:6]
        iv = hashlib.md5(iv_seed).digest()
        encrypted = view[6:]
    else:
        encrypted = view[2:]

    # Decrypt the data
    cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv))
    decryptor = cipher.decryptor()
    decrypted = decryptor.update(encrypted) + decryptor.finalize()

    # Use the length prefix to strip off the correct amount of padding
    return decrypted[:data_len]


def aes_encrypt(data: bytes, aes_key: bytes, iv: Optional[bytes] = None) -> bytes:
    """Encrypt data using non-standard AES-CBC.

    Format: [length:2][iv_seed:4?][encrypted_data:N]

    Args:
        data: Plaintext data to encrypt
        aes_key: 16-byte AES key
        iv: Optional 16-byte IV. If None, random IV seed will be generated

    Returns:
        Encrypted message
    """
    message_header = struct.pack("!H", len(data))

    # If no IV is given, generate a random seed, hash that for the iv, and add
    # the seed to the header
    if iv is None:
        iv_seed = os.urandom(4)
        iv = hashlib.md5(iv_seed).digest()
        message_header += iv_seed

    # Encrypt the data
    aes = algorithms.AES(aes_key)
    padder = PKCS7(aes.block_size).padder()
    padded = padder.update(data) + padder.finalize()
    cipher = Cipher(aes, modes.CBC(iv))
    encryptor = cipher.encryptor()
    encrypted = encryptor.update(padded) + encryptor.finalize()

    return message_header + encrypted


@dataclass
class KeyBundle:
    """Bundle of cryptographic keys needed for encrypted communication.

    Attributes:
        signing_key: EC private key for signing our own public key during handshake
        verify_key: EC public key for verifying peer's signature
        shared_secret: 16-byte static shared secret (same for client and server)
    """

    signing_key: ec.EllipticCurvePrivateKey
    verify_key: ec.EllipticCurvePublicKey
    shared_secret: bytes

    def __post_init__(self):
        if len(self.shared_secret) != 16:
            raise ValueError("shared_secret must be exactly 16 bytes")

    @classmethod
    def from_hex(
        cls, signing_key: str, verify_key: str, shared_secret: str
    ) -> "KeyBundle":
        """Create KeyBundle from hex-encoded strings (for YAML config).

        Args:
            signing_key: EC private key as 64 hex characters
            verify_key: EC public key in DER format as hex
            shared_secret: 16-byte secret as 32 hex characters

        Returns:
            KeyBundle with parsed cryptographic objects

        Raises:
            ValueError: If hex format or key lengths are invalid
        """
        # Validate lengths
        if len(signing_key) != 64:
            raise ValueError("signing_key must be 64 hex characters")
        if len(shared_secret) != 32:
            raise ValueError("shared_secret must be 32 hex characters")

        # Convert hex to crypto objects
        try:
            ec_signing_key = ec.derive_private_key(
                int(signing_key, 16), curve=ec.SECP256R1()
            )
            ec_verify_key = serialization.load_der_public_key(bytes.fromhex(verify_key))
            secret_bytes = bytes.fromhex(shared_secret)
        except ValueError as e:
            raise ValueError(f"Invalid hex in key config: {e}")

        return cls(
            ec_signing_key, cast(ec.EllipticCurvePublicKey, ec_verify_key), secret_bytes
        )


@unique
class HandshakeState(IntEnum):
    CHALLENGE = 1  # server message
    CHALLENGE_RESPONSE = 2  # client message
    CHALLENGE_ACCEPTED = 3  # server message
    SERVER_PUBLIC_KEY = 4  # server message
    CLIENT_PUBLIC_KEY = 5  # client message
    ECDH_ACCEPTED = 6  # server message


class HandshakeMessage:
    PREFIX = b"**"

    def __init__(self, state: HandshakeState, body: bytes) -> None:
        self.state = state
        self.body = body

    def __eq__(self, other):
        if not isinstance(other, HandshakeMessage):
            return False
        return self.state == other.state and self.body == other.body

    def __hash__(self):
        return hash((self.state, self.body))

    def bytearray(self) -> bytearray:
        msg = bytearray(len(self.body) + 6)
        msg[0:2] = self.PREFIX
        struct.pack_into("BB", msg, 2, self.state, len(self.body))
        msg[4:-2] = self.body
        struct.pack_into("!H", msg, -2, sum(msg[2:]))
        return msg

    @classmethod
    def parse(cls, data: bytes) -> Self:
        view = memoryview(data)

        if view[0:2] != cls.PREFIX:
            raise ValueError(f"Prefix is not correct: {view[0:2]}")

        checksum = struct.pack("!H", sum(view[2:-2]))
        if view[-2:] != checksum:
            raise ValueError(f"Checksum is not correct: {checksum!r} vs {view[-2:]}")

        state, length = struct.unpack("BB", view[2:4])
        body_length = len(view) - 6
        if body_length != length:
            raise ValueError(f"Body length should be {length} but is {body_length}")

        return cls(HandshakeState(state), view[4:-2].tobytes())


class HandshakeProtocol:
    def __init__(self, key_bundle: KeyBundle) -> None:
        self._key_bundle = key_bundle
        self._aes_key: Optional[bytes] = None
        self._aes_iv: Optional[bytes] = None
        self._ephemeral_private_key: Optional[ec.EllipticCurvePrivateKey] = None
        self._ephemeral_public_key: Optional[ec.EllipticCurvePublicKey] = None
        self._peer_public_key: Optional[ec.EllipticCurvePublicKey] = None
        self._session_aes_key: Optional[bytes] = None

    @property
    def session_aes_key(self):
        """The AES key to use after the handshake has been completed"""
        return self._session_aes_key

    def handle(self, data: Optional[bytes]) -> Optional[bytearray]:
        """Decodes client or server messages and generates the correct response.

        Attributes:
            data: The preceeding message or None for the initial server message
                  for the current state
        """

        if data is None:
            # Handle server-initiated rounds - challenge then key exchange
            if self._aes_key is None:
                response = self._generate_challenge()
            else:
                response = self._generate_server_public_key()
        else:
            # Parse message
            if self._ephemeral_public_key and self._aes_key and self._aes_iv:
                decrypted = aes_decrypt(data, self._aes_key, self._aes_iv)
                message = HandshakeMessage.parse(decrypted)
            else:
                message = HandshakeMessage.parse(data)

            # Handle message responses
            if message.state == HandshakeState.CHALLENGE:
                response = self._handle_challenge(message)
            elif message.state == HandshakeState.CHALLENGE_RESPONSE:
                response = self._handle_challenge_response(message)
            elif message.state == HandshakeState.CHALLENGE_ACCEPTED:
                self._handle_challenge_accepted(message)
                return None
            elif message.state == HandshakeState.SERVER_PUBLIC_KEY:
                response = self._handle_server_public_key(message)
            elif message.state == HandshakeState.CLIENT_PUBLIC_KEY:
                response = self._handle_client_public_key(message)
            elif message.state == HandshakeState.ECDH_ACCEPTED:
                self._handle_ecdh_accepted(message)
                return None

        # Convert response to a bytearray
        if self._ephemeral_public_key and self._aes_key and self._aes_iv:
            response_bytes = bytes(response.bytearray())
            encrypted_bytes = aes_encrypt(response_bytes, self._aes_key, self._aes_iv)
            return bytearray(encrypted_bytes)
        elif response:
            return response.bytearray()
        else:
            raise Exception("This should not be possible")

    def _generate_challenge(self) -> HandshakeMessage:
        challenge = os.urandom(4)
        self._set_aes_key(challenge)
        return HandshakeMessage(HandshakeState.CHALLENGE, challenge)

    def _handle_challenge(self, message: HandshakeMessage) -> HandshakeMessage:
        self._set_aes_key(message.body)
        assert self._aes_iv is not None
        return HandshakeMessage(HandshakeState.CHALLENGE_RESPONSE, self._aes_iv[8:12])

    def _handle_challenge_response(self, message: HandshakeMessage) -> HandshakeMessage:
        assert self._aes_iv is not None
        if message.body == self._aes_iv[8:12]:
            return HandshakeMessage(HandshakeState.CHALLENGE_ACCEPTED, b"\x00")
        else:
            return HandshakeMessage(HandshakeState.CHALLENGE_ACCEPTED, b"\x01")

    def _handle_challenge_accepted(self, message: HandshakeMessage) -> None:
        if message.body != b"\x00":
            raise ValueError(f"Challenge response was not 0: {message.body!r}")

        # Generate ephemeral keys so that future messages are assumed to be
        # encrypted
        self._set_ephemeral_keys()

    def _generate_server_public_key(self) -> HandshakeMessage:
        self._set_ephemeral_keys()
        key_and_signature = self._sign_public_key()
        return HandshakeMessage(HandshakeState.SERVER_PUBLIC_KEY, key_and_signature)

    def _handle_server_public_key(self, message: HandshakeMessage) -> HandshakeMessage:
        # Verify and load the server public key
        self._peer_public_key = self._verify_public_key(message.body)

        # Build response message
        key_and_signature = self._sign_public_key()
        return HandshakeMessage(HandshakeState.CLIENT_PUBLIC_KEY, key_and_signature)

    def _handle_client_public_key(self, message: HandshakeMessage) -> HandshakeMessage:
        # Verify and load the client public key
        self._peer_public_key = self._verify_public_key(message.body)

        # Set the session aes key
        assert self._ephemeral_private_key is not None
        self._session_aes_key = self._ephemeral_private_key.exchange(
            ec.ECDH(), self._peer_public_key
        )

        # Build response message
        return HandshakeMessage(HandshakeState.ECDH_ACCEPTED, b"\x00")

    def _handle_ecdh_accepted(self, message: HandshakeMessage) -> None:
        if message.body != b"\x00":
            raise ValueError(f"ECDH accepted response was not 0: {message.body!r}")

        # Set the session aes key
        assert self._ephemeral_private_key is not None
        assert self._peer_public_key is not None
        self._session_aes_key = self._ephemeral_private_key.exchange(
            ec.ECDH(), self._peer_public_key
        )

    def _sign_public_key(self) -> bytes:
        assert self._aes_iv is not None
        assert self._ephemeral_public_key is not None

        # Convert the ephemeral public key to a 64 byte string, removing the
        # first byte that says that it's an uncompressed point
        public_key_bytes = self._ephemeral_public_key.public_bytes(
            serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint
        )[1:]

        # Sign the public key with our signing key and convert the DER-encoded
        # signature to a raw 64 byte string
        signing_data = public_key_bytes + self._aes_iv
        der_signature = self._key_bundle.signing_key.sign(
            signing_data, ec.ECDSA(hashes.SHA256())
        )
        r, s = decode_dss_signature(der_signature)
        signature = r.to_bytes(32, "big") + s.to_bytes(32, "big")

        return public_key_bytes + signature

    def _verify_public_key(self, key_and_signature: bytes) -> ec.EllipticCurvePublicKey:
        assert self._aes_iv is not None

        # Split the message into the server public key and the signature
        if len(key_and_signature) != 128:
            raise ValueError(
                f"Signed key length should be 128 but is {len(key_and_signature)}"
            )
        data, signature = key_and_signature[:64], key_and_signature[64:]

        # Check the signature - it needs to be DER-encoded first
        signing_data = data + self._aes_iv
        r, s = int.from_bytes(signature[:32], "big"), int.from_bytes(
            signature[32:], "big"
        )
        der_signature = encode_dss_signature(r, s)
        self._key_bundle.verify_key.verify(
            der_signature, signing_data, ec.ECDSA(hashes.SHA256())
        )

        # Return public key
        return ec.EllipticCurvePublicKey.from_encoded_point(
            ec.SECP256R1(), b"\x04" + data
        )

    def _set_aes_key(self, challenge: bytes) -> None:
        if len(challenge) != 4:
            raise ValueError(
                f"Expected challenge to be 4 bytes but was {len(challenge)}"
            )

        # Calculate MD5 of reverse order challenge for IV
        self._aes_iv = hashlib.md5(challenge[::-1]).digest()

        # XOR the IV with the shared secret to derive the key
        self._aes_key = bytes(
            [x ^ y for x, y in zip(self._aes_iv, self._key_bundle.shared_secret)]
        )

    def _set_ephemeral_keys(self) -> None:
        self._ephemeral_private_key = ec.generate_private_key(ec.SECP256R1())
        self._ephemeral_public_key = self._ephemeral_private_key.public_key()
