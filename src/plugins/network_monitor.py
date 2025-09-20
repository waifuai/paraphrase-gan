"""
Network Monitor Plugin

This plugin provides network monitoring and diagnostic capabilities including:
- Network connectivity testing
- DNS resolution
- Port scanning
- Bandwidth monitoring
- Network interface monitoring
"""

import socket
import subprocess
import time
from typing import Dict, List, Optional, Union
from datetime import datetime
import json
import os
import platform
import requests
import psutil


class NetworkMonitor:
    """Network monitoring and diagnostic tools."""

    def __init__(self):
        self.system = platform.system().lower()

    def ping_host(self, host: str, count: int = 4, timeout: int = 5) -> Dict[str, Union[bool, str, float, list]]:
        """Ping a host and return statistics."""
        try:
            if self.system == "windows":
                cmd = ["ping", "-n", str(count), "-w", str(timeout * 1000), host]
            else:
                cmd = ["ping", "-c", str(count), "-W", str(timeout), host]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout * count + 5)

            return {
                "success": result.returncode == 0,
                "host": host,
                "return_code": result.returncode,
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip()
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "host": host, "error": "Timeout expired"}
        except Exception as e:
            return {"success": False, "host": host, "error": str(e)}

    def resolve_dns(self, hostname: str, record_type: str = 'A') -> Dict[str, Union[bool, str, list]]:
        """Resolve DNS records for a hostname."""
        try:
            if record_type.upper() == 'A':
                results = socket.getaddrinfo(hostname, None, socket.AF_INET)
                addresses = list(set([result[4][0] for result in results]))
            elif record_type.upper() == 'AAAA':
                results = socket.getaddrinfo(hostname, None, socket.AF_INET6)
                addresses = list(set([result[4][0] for result in results]))
            else:
                return {"success": False, "hostname": hostname, "error": f"Unsupported record type: {record_type}"}

            return {
                "success": True,
                "hostname": hostname,
                "record_type": record_type,
                "addresses": addresses
            }
        except socket.gaierror as e:
            return {"success": False, "hostname": hostname, "error": str(e)}
        except Exception as e:
            return {"success": False, "hostname": hostname, "error": str(e)}

    def check_port(self, host: str, port: int, timeout: float = 5) -> Dict[str, Union[bool, str, int, float]]:
        """Check if a port is open on a host."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            sock.close()

            is_open = result == 0
            return {
                "success": True,
                "host": host,
                "port": port,
                "is_open": is_open,
                "status": "open" if is_open else "closed"
            }
        except Exception as e:
            return {"success": False, "host": host, "port": port, "error": str(e)}

    def scan_ports(self, host: str, ports: List[int], timeout: float = 1) -> List[Dict[str, Union[str, int, bool]]]:
        """Scan multiple ports on a host."""
        results = []
        for port in ports:
            result = self.check_port(host, port, timeout)
            results.append(result)
        return results

    def get_network_interfaces(self) -> Dict[str, Dict[str, Union[str, list, bool]]]:
        """Get detailed network interface information."""
        interfaces = {}

        # Get interface addresses
        for interface, addrs in psutil.net_if_addrs().items():
            interfaces[interface] = {
                "addresses": [],
                "status": "up"
            }

            for addr in addrs:
                if addr.family == socket.AF_INET:
                    interfaces[interface]["addresses"].append({
                        "family": "IPv4",
                        "address": addr.address,
                        "netmask": addr.netmask,
                        "broadcast": addr.broadcast
                    })
                elif addr.family == socket.AF_INET6:
                    interfaces[interface]["addresses"].append({
                        "family": "IPv6",
                        "address": addr.address,
                        "netmask": addr.netmask
                    })
                elif addr.family == socket.AF_PACKET:
                    interfaces[interface]["addresses"].append({
                        "family": "MAC",
                        "address": addr.address,
                        "netmask": addr.netmask,
                        "broadcast": addr.broadcast
                    })

        # Get interface statistics
        if_stats = psutil.net_if_stats()
        for interface in interfaces:
            if interface in if_stats:
                stats = if_stats[interface]
                interfaces[interface]["status"] = "up" if stats.isup else "down"
                interfaces[interface]["mtu"] = stats.mtu
                interfaces[interface]["speed"] = getattr(stats, 'speed', None)

        return interfaces

    def test_connectivity(self, targets: List[str] = None) -> Dict[str, Dict[str, Union[bool, str, float]]]:
        """Test connectivity to multiple targets."""
        if targets is None:
            targets = ["8.8.8.8", "1.1.1.1", "google.com"]

        results = {}
        for target in targets:
            start_time = time.time()
            try:
                # Try to resolve DNS first
                socket.gethostbyname(target)
                dns_success = True
                dns_time = time.time() - start_time
            except socket.gaierror:
                dns_success = False
                dns_time = None

            # Try to ping
            ping_result = self.ping_host(target, count=2, timeout=2)
            ping_success = ping_result["success"]
            ping_time = time.time() - start_time if dns_success else None

            results[target] = {
                "dns_resolution": dns_success,
                "dns_time": dns_time,
                "ping_success": ping_success,
                "total_time": time.time() - start_time,
                "status": "reachable" if ping_success else "unreachable"
            }

        return results

    def get_network_connections(self) -> List[Dict[str, Union[str, int]]]:
        """Get current network connections."""
        connections = []
        for conn in psutil.net_connections():
            connections.append({
                "fd": conn.fd,
                "family": str(conn.family),
                "type": str(conn.type),
                "local_address": f"{conn.laddr.ip}:{conn.laddr.port}" if conn.laddr else None,
                "remote_address": f"{conn.raddr.ip}:{conn.raddr.port}" if conn.raddr else None,
                "status": conn.status,
                "pid": conn.pid
            })
        return connections

    def traceroute(self, host: str, max_hops: int = 30) -> Dict[str, Union[bool, str, list]]:
        """Perform traceroute to a host."""
        try:
            if self.system == "windows":
                cmd = ["tracert", "-h", str(max_hops), "-w", "2000", host]
            else:
                cmd = ["traceroute", "-m", str(max_hops), "-w", "2", host]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            return {
                "success": result.returncode == 0,
                "host": host,
                "return_code": result.returncode,
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
                "hops": result.stdout.strip().split('\n') if result.returncode == 0 else []
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "host": host, "error": "Traceroute timed out"}
        except Exception as e:
            return {"success": False, "host": host, "error": str(e)}

    def get_network_stats(self) -> Dict[str, Dict[str, Union[int, float]]]:
        """Get network I/O statistics."""
        net_io = psutil.net_io_counters(pernic=True)
        stats = {}

        for interface, counters in net_io.items():
            stats[interface] = {
                "bytes_sent": counters.bytes_sent,
                "bytes_recv": counters.bytes_recv,
                "packets_sent": counters.packets_sent,
                "packets_recv": counters.packets_recv,
                "errin": counters.errin,
                "errout": counters.errout,
                "dropin": counters.dropin,
                "dropout": counters.dropout
            }

        return stats

    def check_http_endpoint(self, url: str, timeout: int = 10) -> Dict[str, Union[bool, str, int, float]]:
        """Check HTTP endpoint availability."""
        try:
            start_time = time.time()
            response = requests.get(url, timeout=timeout, allow_redirects=True)
            response_time = time.time() - start_time

            return {
                "success": True,
                "url": url,
                "status_code": response.status_code,
                "response_time": round(response_time * 1000, 2),  # ms
                "content_length": len(response.content),
                "headers": dict(response.headers),
                "is_redirect": len(response.history) > 0,
                "final_url": response.url
            }
        except requests.exceptions.Timeout:
            return {"success": False, "url": url, "error": "Request timed out"}
        except requests.exceptions.ConnectionError:
            return {"success": False, "url": url, "error": "Connection error"}
        except requests.exceptions.RequestException as e:
            return {"success": False, "url": url, "error": str(e)}
        except Exception as e:
            return {"success": False, "url": url, "error": str(e)}


# Plugin registration
def register_plugin():
    """Register the network monitor plugin."""
    return {
        "name": "network_monitor",
        "description": "Network monitoring and diagnostic tools",
        "version": "1.0.0",
        "class": NetworkMonitor,
        "methods": [
            "ping_host",
            "resolve_dns",
            "check_port",
            "scan_ports",
            "get_network_interfaces",
            "test_connectivity",
            "get_network_connections",
            "traceroute",
            "get_network_stats",
            "check_http_endpoint"
        ]
    }