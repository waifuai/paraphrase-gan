"""
Plugin System

This module provides a plugin system for extending the application's functionality.
"""

import importlib
import os
from typing import Dict, List, Any, Optional
from pathlib import Path


class PluginManager:
    """Manages loading and execution of plugins."""

    def __init__(self, plugins_dir: Optional[str] = None):
        if plugins_dir is None:
            # Default to the directory containing this file
            plugins_dir = Path(__file__).parent
        else:
            plugins_dir = Path(plugins_dir)

        self.plugins_dir = plugins_dir
        self.loaded_plugins = {}
        self.plugin_info = {}

    def discover_plugins(self) -> List[str]:
        """Discover available plugins in the plugins directory."""
        plugins = []

        if not self.plugins_dir.exists():
            return plugins

        for file in self.plugins_dir.glob("*.py"):
            if file.name != "__init__.py" and not file.name.startswith("_"):
                plugin_name = file.stem
                plugins.append(plugin_name)

        return plugins

    def load_plugin(self, plugin_name: str) -> bool:
        """Load a specific plugin."""
        try:
            module_path = f"src.plugins.{plugin_name}"
            module = importlib.import_module(module_path)

            if not hasattr(module, 'register_plugin'):
                print(f"Plugin {plugin_name} does not have register_plugin function")
                return False

            plugin_info = module.register_plugin()
            plugin_class = plugin_info['class']

            # Instantiate the plugin class
            plugin_instance = plugin_class()

            self.loaded_plugins[plugin_name] = plugin_instance
            self.plugin_info[plugin_name] = plugin_info

            print(f"Successfully loaded plugin: {plugin_name}")
            return True

        except Exception as e:
            print(f"Failed to load plugin {plugin_name}: {str(e)}")
            return False

    def load_all_plugins(self) -> Dict[str, bool]:
        """Load all discovered plugins."""
        plugins = self.discover_plugins()
        results = {}

        for plugin_name in plugins:
            results[plugin_name] = self.load_plugin(plugin_name)

        return results

    def get_plugin(self, plugin_name: str) -> Optional[Any]:
        """Get a loaded plugin instance."""
        return self.loaded_plugins.get(plugin_name)

    def list_loaded_plugins(self) -> Dict[str, Dict[str, Any]]:
        """List all loaded plugins with their information."""
        return self.plugin_info.copy()

    def execute_plugin_method(self, plugin_name: str, method_name: str, *args, **kwargs) -> Any:
        """Execute a method on a specific plugin."""
        plugin = self.get_plugin(plugin_name)
        if not plugin:
            raise ValueError(f"Plugin '{plugin_name}' not loaded")

        if not hasattr(plugin, method_name):
            raise AttributeError(f"Plugin '{plugin_name}' has no method '{method_name}'")

        method = getattr(plugin, method_name)
        return method(*args, **kwargs)


# Global plugin manager instance
plugin_manager = PluginManager()


def initialize_plugins() -> PluginManager:
    """Initialize and load all available plugins."""
    results = plugin_manager.load_all_plugins()

    print(f"\nPlugin Loading Summary:")
    print(f"{'Plugin':<20} {'Status':<10} {'Version':<10} {'Description'}")
    print("-" * 80)

    for plugin_name, loaded in results.items():
        if loaded:
            info = plugin_manager.plugin_info.get(plugin_name, {})
            version = info.get('version', 'N/A')
            description = info.get('description', 'N/A')
            print(f"{plugin_name:<20} {'✓ Loaded':<10} {version:<10} {description}")
        else:
            print(f"{plugin_name:<20} {'✗ Failed':<10} {'N/A':<10} N/A")

    print()
    return plugin_manager


# Convenience functions for easy access
def get_plugin_manager() -> PluginManager:
    """Get the global plugin manager instance."""
    return plugin_manager


def use_plugin(plugin_name: str, method_name: str, *args, **kwargs) -> Any:
    """Convenience function to use a plugin method."""
    return plugin_manager.execute_plugin_method(plugin_name, method_name, *args, **kwargs)