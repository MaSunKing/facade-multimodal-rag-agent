"""Application-only VPN Fake-IP compatibility; never change system settings."""
import ipaddress
import json
import os
import socket
import threading
import time
from pathlib import Path
from urllib.parse import urlencode, urlsplit
from urllib.request import ProxyHandler, build_opener

_CACHE = {}
_LOCK = threading.Lock()


def configured_local_proxy():
    value = os.getenv('WEB_FETCH_LOCAL_PROXY', '')
    if not value:
        path = Path(__file__).resolve().parents[2] / 'runtime' / 'web_fetch.local.json'
        if path.exists():
            value = str(json.loads(path.read_text(encoding='utf-8')).get('local_proxy') or '')
    if not value:
        return None
    parts = urlsplit(value)
    if parts.scheme != 'http' or parts.hostname not in {'127.0.0.1', 'localhost', '::1'} or not parts.port:
        raise ValueError('web_proxy_must_be_explicit_loopback_http_proxy')
    if parts.username or parts.password:
        raise ValueError('web_proxy_credentials_not_supported')
    return value


def resolve_public_host(host, port):
    addresses = list(dict.fromkeys(a[4][0] for a in socket.getaddrinfo(host, port)))
    ips = [ipaddress.ip_address(a) for a in addresses]
    fake = [ip for ip in ips if ip.version == 4 and ip in ipaddress.ip_network('198.18.0.0/15')]
    # Private/loopback/link-local entries must NEVER be laundered via public DNS.
    if any(not ip.is_global and ip not in fake for ip in ips):
        raise ValueError('non_public_address')
    if not fake:
        return addresses, None
    proxy = configured_local_proxy()
    if not proxy:
        raise ValueError('proxy_fake_ip_requires_local_proxy')
    key = (host, proxy)
    with _LOCK:
        cached = _CACHE.get(key)
    if cached and cached[0] > time.monotonic():
        return cached[1], proxy
    # Only the public hostname goes to a fixed HTTPS resolver through the
    # explicitly configured local proxy. TLS verification remains enabled.
    url = 'https://dns.alidns.com/resolve?' + urlencode({'name':host, 'type':'A'})
    opener = build_opener(ProxyHandler({'http':proxy, 'https':proxy}))
    with opener.open(url, timeout=5) as response:
        if urlsplit(response.geturl()).hostname != 'dns.alidns.com':
            raise ValueError('public_dns_unexpected_redirect')
        body = response.read(65537)
        if len(body)>65536:
            raise ValueError('public_dns_response_too_large')
        data = json.loads(body)
    public = [str(a.get('data')) for a in data.get('Answer', []) if a.get('type') == 1]
    if not public or any(not ipaddress.ip_address(a).is_global for a in public):
        raise ValueError('public_dns_no_safe_address')
    with _LOCK:
        _CACHE[key] = (time.monotonic()+60, public)
    return public, proxy
