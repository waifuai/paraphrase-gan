"""
System Admin Tools Plugin

This plugin provides comprehensive system administration capabilities for the application.
It extends the core functionality with tools for process monitoring and management,
system resource monitoring, service management, and network monitoring.

The plugin includes methods for retrieving system information, monitoring CPU, memory,
and disk usage, managing processes, checking network interfaces, and performing
service management operations on Linux systems.

This plugin provides system administration capabilities including:
- Process monitoring and management
- System resource monitoring
- Service management
- Network monitoring
"""

import psutil
import subprocess
import platform
from typing import Dict, List, Optional, Union
from datetime import datetime
import json
import os
import signal


class SystemAdminTools:
    """System administration tools for monitoring and managing system resources."""

    def __init__(self):
        self.system = platform.system().lower()

    def get_system_info(self) -> Dict[str, Union[str, float, int]]:
        """Get basic system information."""
        return {
            "system": platform.system(),
            "node": platform.node(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total,
            "boot_time": datetime.fromtimestamp(psutil.boot_time()).isoformat()
        }

    def get_cpu_info(self) -> Dict[str, Union[float, int]]:
        """Get CPU information and usage."""
        return {
            "percent": psutil.cpu_percent(interval=1),
            "count": psutil.cpu_count(),
            "count_logical": psutil.cpu_count(logical=True),
            "freq_current": psutil.cpu_freq().current if psutil.cpu_freq() else None,
            "freq_min": psutil.cpu_freq().min if psutil.cpu_freq() else None,
            "freq_max": psutil.cpu_freq().max if psutil.cpu_freq() else None,
            "load_avg": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
        }

    def get_memory_info(self) -> Dict[str, Union[float, int]]:
        """Get memory information."""
        mem = psutil.virtual_memory()
        return {
            "total": mem.total,
            "available": mem.available,
            "percent": mem.percent,
            "used": mem.used,
            "free": mem.free,
            "active": getattr(mem, 'active', None),
            "inactive": getattr(mem, 'inactive', None),
            "buffers": getattr(mem, 'buffers', None),
            "cached": getattr(mem, 'cached', None),
            "shared": getattr(mem, 'shared', None)
        }

    def get_disk_info(self) -> Dict[str, Dict[str, Union[float, int]]]:
        """Get disk usage information."""
        disk_info = {}
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                disk_info[partition.device] = {
                    "mountpoint": partition.mountpoint,
                    "filesystem": partition.fstype,
                    "total": usage.total,
                    "used": usage.used,
                    "free": usage.free,
                    "percent": usage.percent
                }
            except PermissionError:
                continue
        return disk_info

    def get_network_info(self) -> Dict[str, Dict[str, Union[float, int, str]]]:
        """Get network interface information."""
        network_info = {}
        for interface, addrs in psutil.net_if_addrs().items():
            network_info[interface] = {
                "addresses": [str(addr.address) for addr in addrs],
                "status": "up" if interface in psutil.net_if_stats() and
                         psutil.net_if_stats()[interface].isup else "down"
            }
        return network_info

    def get_processes(self, limit: int = 10) -> List[Dict[str, Union[int, str, float]]]:
        """Get list of running processes."""
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'username', 'cpu_percent',
                                       'memory_percent', 'status', 'create_time']):
            try:
                processes.append({
                    "pid": proc.info['pid'],
                    "name": proc.info['name'],
                    "username": proc.info['username'],
                    "cpu_percent": round(proc.info['cpu_percent'], 2),
                    "memory_percent": round(proc.info['memory_percent'], 2),
                    "status": proc.info['status'],
                    "create_time": datetime.fromtimestamp(proc.info['create_time']).isoformat()
                })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        # Sort by CPU usage and limit results
        processes.sort(key=lambda x: x['cpu_percent'], reverse=True)
        return processes[:limit]

    def kill_process(self, pid: int, signal_type: int = signal.SIGTERM) -> Dict[str, Union[bool, str]]:
        """Kill a process by PID."""
        try:
            process = psutil.Process(pid)
            process.send_signal(signal_type)
            return {"success": True, "message": f"Signal {signal_type} sent to process {pid}"}
        except psutil.NoSuchProcess:
            return {"success": False, "message": f"Process {pid} not found"}
        except psutil.AccessDenied:
            return {"success": False, "message": f"Access denied to process {pid}"}
        except Exception as e:
            return {"success": False, "message": str(e)}

    def get_process_details(self, pid: int) -> Optional[Dict[str, Union[int, str, float, list]]]:
        """Get detailed information about a specific process."""
        try:
            process = psutil.Process(pid)
            return {
                "pid": process.pid,
                "name": process.name(),
                "exe": process.exe(),
                "cmdline": process.cmdline(),
                "username": process.username(),
                "status": process.status(),
                "cpu_percent": process.cpu_percent(),
                "memory_percent": process.memory_percent(),
                "memory_info": dict(process.memory_info()._asdict()),
                "create_time": datetime.fromtimestamp(process.create_time()).isoformat(),
                "num_threads": process.num_threads(),
                "open_files": [str(f.path) for f in process.open_files()] if process.open_files() else []
            }
        except psutil.NoSuchProcess:
            return None
        except psutil.AccessDenied:
            return {"error": "Access denied"}
        except Exception as e:
            return {"error": str(e)}

    def manage_service(self, service_name: str, action: str) -> Dict[str, Union[bool, str]]:
        """Manage system services (Linux only)."""
        if self.system not in ['linux']:
            return {"success": False, "message": "Service management only supported on Linux"}

        valid_actions = ['start', 'stop', 'restart', 'status']
        if action not in valid_actions:
            return {"success": False, "message": f"Invalid action. Must be one of {valid_actions}"}

        try:
            if action == 'status':
                result = subprocess.run(['systemctl', 'is-active', service_name],
                                      capture_output=True, text=True, timeout=10)
                is_active = result.returncode == 0
                return {
                    "success": True,
                    "service": service_name,
                    "action": action,
                    "is_active": is_active,
                    "output": result.stdout.strip()
                }
            else:
                result = subprocess.run(['systemctl', action, service_name],
                                      capture_output=True, text=True, timeout=30)
                success = result.returncode == 0
                return {
                    "success": success,
                    "service": service_name,
                    "action": action,
                    "output": result.stdout.strip(),
                    "error": result.stderr.strip() if not success else None
                }
        except subprocess.TimeoutExpired:
            return {"success": False, "message": f"Command timed out for service {service_name}"}
        except FileNotFoundError:
            return {"success": False, "message": "systemctl command not found"}
        except Exception as e:
            return {"success": False, "message": str(e)}

    def get_system_health(self) -> Dict[str, Union[str, float, int, bool]]:
        """Get overall system health status."""
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        disk_usage = psutil.disk_usage('/')

        # Determine health status
        cpu_healthy = cpu_percent < 90
        memory_healthy = memory.percent < 90
        disk_healthy = disk_usage.percent < 90

        overall_healthy = cpu_healthy and memory_healthy and disk_healthy

        return {
            "timestamp": datetime.now().isoformat(),
            "overall_healthy": overall_healthy,
            "cpu": {
                "percent": cpu_percent,
                "healthy": cpu_healthy
            },
            "memory": {
                "percent": memory.percent,
                "healthy": memory_healthy
            },
            "disk": {
                "percent": disk_usage.percent,
                "healthy": disk_healthy
            }
        }


# Plugin registration
def register_plugin():
    """Register the system admin tools plugin."""
    return {
        "name": "system_admin",
        "description": "System administration tools for monitoring and managing system resources",
        "version": "1.0.0",
        "class": SystemAdminTools,
        "methods": [
            "get_system_info",
            "get_cpu_info",
            "get_memory_info",
            "get_disk_info",
            "get_network_info",
            "get_processes",
            "kill_process",
            "get_process_details",
            "manage_service",
            "get_system_health"
        ]
    }