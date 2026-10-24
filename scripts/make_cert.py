"""Create a self-signed TLS cert so phones on the LAN can use the camera (HTTPS required).

Usage: uv run python scripts/make_cert.py [extra-ip-or-host ...]
"""

import ipaddress
import socket
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.config import get_settings  # noqa: E402


def _lan_ip() -> str | None:
    # No packets are sent; this just asks the OS which interface routes outward.
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
        except OSError:
            return None


def _san_entry(name: str) -> str:
    try:
        ipaddress.ip_address(name)
        return f"IP:{name}"
    except ValueError:
        return f"DNS:{name}"


def main() -> None:
    settings = get_settings()
    names = ["localhost", "127.0.0.1", socket.gethostname(), *sys.argv[1:]]
    if ip := _lan_ip():
        names.append(ip)
    san = ",".join(dict.fromkeys(_san_entry(n) for n in names))

    settings.ssl_certfile.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "825",
            "-keyout", str(settings.ssl_keyfile), "-out", str(settings.ssl_certfile),
            "-subj", "/CN=face-attendance", "-addext", f"subjectAltName={san}",
        ],
        check=True,
    )
    print(f"✓ {settings.ssl_certfile}\n  SAN: {san}")
    if ip:
        url = f"https://{ip}:{settings.server_port}"
        print(f"  Mở trên điện thoại: {url} (chấp nhận cảnh báo cert)")


if __name__ == "__main__":
    main()
