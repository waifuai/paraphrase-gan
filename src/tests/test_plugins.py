"""
Tests for plugin system and individual plugins.
"""

import pytest
import pandas as pd
import tempfile
import os
from unittest.mock import Mock, patch

from src.plugins import get_plugin_manager, initialize_plugins
from src.plugins.system_admin import SystemAdminTools
from src.plugins.network_monitor import NetworkMonitor
from src.plugins.data_analysis import DataAnalysis


def test_plugin_manager():
    """Test plugin manager functionality."""
    manager = get_plugin_manager()

    assert manager is not None

    # Test plugin discovery
    plugins = manager.discover_plugins()
    assert isinstance(plugins, list)

    # Should find our plugins
    expected_plugins = ['system_admin', 'network_monitor', 'data_analysis']
    for plugin in expected_plugins:
        assert plugin in plugins


def test_system_admin_plugin():
    """Test system admin plugin functionality."""
    plugin = SystemAdminTools()

    # Test system info
    info = plugin.get_system_info()
    assert isinstance(info, dict)
    assert 'system' in info
    assert 'cpu_count' in info

    # Test CPU info
    cpu_info = plugin.get_cpu_info()
    assert isinstance(cpu_info, dict)
    assert 'percent' in cpu_info

    # Test memory info
    mem_info = plugin.get_memory_info()
    assert isinstance(mem_info, dict)
    assert 'total' in mem_info
    assert 'available' in mem_info

    # Test process listing (should work without errors)
    processes = plugin.get_processes(limit=5)
    assert isinstance(processes, list)
    assert len(processes) <= 5


def test_network_monitor_plugin():
    """Test network monitor plugin functionality."""
    plugin = NetworkMonitor()

    # Test connectivity check
    connectivity = plugin.test_connectivity(['8.8.8.8'])
    assert isinstance(connectivity, dict)

    # Test DNS resolution
    dns_result = plugin.resolve_dns('google.com')
    assert isinstance(dns_result, dict)
    assert 'success' in dns_result

    # Test port checking
    port_result = plugin.check_port('8.8.8.8', 53)
    assert isinstance(port_result, dict)
    assert 'is_open' in port_result


def test_data_analysis_plugin():
    """Test data analysis plugin functionality."""
    plugin = DataAnalysis()

    # Test with sample data
    test_data = pd.DataFrame({
        'name': ['Alice', 'Bob', 'Charlie'],
        'age': [25, 30, 35],
        'city': ['New York', 'London', 'Tokyo']
    })

    # Test data info
    info = plugin.get_data_info(test_data)
    assert info['success'] is True
    assert 'shape' in info['info']
    assert info['info']['shape'] == (3, 3)

    # Test statistics calculation
    stats = plugin.calculate_statistics(test_data, 'age')
    assert stats['success'] is True
    assert 'mean' in stats['statistics']
    assert stats['statistics']['mean'] == 30.0

    # Test filtering
    filtered = plugin.filter_data(test_data, [
        {'column': 'age', 'operator': '>', 'value': 25}
    ])
    assert filtered['success'] is True
    assert len(filtered['data']) == 2


def test_data_analysis_file_operations():
    """Test data analysis file operations."""
    plugin = DataAnalysis()

    # Create test data
    test_data = pd.DataFrame({
        'id': [1, 2, 3],
        'value': [10, 20, 30]
    })

    with tempfile.TemporaryDirectory() as temp_dir:
        # Test saving data
        csv_path = os.path.join(temp_dir, 'test.csv')
        save_result = plugin.save_data(test_data, csv_path, 'csv')
        assert save_result['success'] is True

        # Test loading data
        load_result = plugin.load_data(csv_path, 'csv')
        assert load_result['success'] is True
        assert len(load_result['data']) == 3
        assert list(load_result['data'].columns) == ['id', 'value']


def test_data_analysis_cleaning():
    """Test data cleaning operations."""
    plugin = DataAnalysis()

    # Create data with issues
    test_data = pd.DataFrame({
        'text': ['  hello  ', '  world  ', '  hello  '],
        'number': [1, 2, None],
        'category': ['A', 'B', 'A']
    })

    # Test cleaning operations
    cleaned = plugin.clean_data(test_data, ['strip_whitespace', 'drop_duplicates'])
    assert cleaned['success'] is True
    assert len(cleaned['data']) == 2  # Should remove duplicate after stripping


def test_data_analysis_aggregation():
    """Test data aggregation functionality."""
    plugin = DataAnalysis()

    test_data = pd.DataFrame({
        'group': ['A', 'A', 'B', 'B'],
        'value': [10, 20, 30, 40],
        'count': [1, 2, 1, 2]
    })

    # Test aggregation
    agg_result = plugin.aggregate_data(
        test_data,
        group_by=['group'],
        aggregations={'value': ['sum', 'mean'], 'count': ['sum']}
    )

    assert agg_result['success'] is True
    assert len(agg_result['data']) == 2  # Two groups
    assert 'value_sum' in agg_result['data'].columns
    assert 'value_mean' in agg_result['data'].columns


@patch('psutil.cpu_percent')
@patch('psutil.virtual_memory')
def test_system_health_monitoring(mock_memory, mock_cpu):
    """Test system health monitoring with mocked values."""
    # Mock high CPU usage and low memory
    mock_cpu.return_value = 95.0
    mock_memory.return_value = Mock(percent=92.0)

    plugin = SystemAdminTools()
    health = plugin.get_system_health()

    assert health['overall_healthy'] is False
    assert health['cpu']['percent'] == 95.0
    assert health['cpu']['healthy'] is False
    assert health['memory']['percent'] == 92.0
    assert health['memory']['healthy'] is False


def test_plugin_initialization():
    """Test plugin system initialization."""
    with patch('src.plugins.PluginManager.load_all_plugins') as mock_load:
        mock_load.return_value = {
            'system_admin': True,
            'network_monitor': True,
            'data_analysis': False
        }

        manager = initialize_plugins()

        assert manager is not None
        mock_load.assert_called_once()