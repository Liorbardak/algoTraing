import json
import os
from typing import Dict, Any, Optional
from pathlib import Path
import os

class ConfigManager:
    """
    Configuration Manager for handling JSON parameters file
    with support for grouped parameters and runtime modifications
    """

    def __init__(self, config_file: str = os.path.join(Path(__file__).parent, "trading_config.json")):
        self.config_file = Path(config_file)
        self.config_data = {}
        self.base_dir = Path(__file__).parent.absolute()  # Directory of the config file
        self.load_config()
        self._resolve_paths()

    def load_config(self) -> None:
        """Load configuration from JSON file"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    self.config_data = json.load(f)
                print(f"✓ Configuration loaded from {self.config_file}")
            else:
                print(f"⚠ Config file not found: {self.config_file}")
                self.config_data = {}
        except json.JSONDecodeError as e:
            print(f"✗ Error parsing JSON: {e}")
            self.config_data = {}
        except Exception as e:
            print(f"✗ Error loading config: {e}")
            self.config_data = {}

    def save_config(self) -> None:
        """Save current configuration to JSON file"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config_data, f, indent=2, ensure_ascii=False)
            print(f"✓ Configuration saved to {self.config_file}")
        except Exception as e:
            print(f"✗ Error saving config: {e}")

    def get_group(self, group_name: str) -> Dict[str, Any]:
        """Get all parameters from a specific group"""
        return self.config_data.get(group_name, {})

    def get_parameter(self, group_name: str, param_name: str, default=None) -> Any:
        """Get a specific parameter from a group"""
        group = self.get_group(group_name)
        return group.get(param_name, default)

    def set_parameter(self, group_name: str, param_name: str, value: Any) -> None:
        """Set a parameter value in a specific group"""
        if group_name not in self.config_data:
            self.config_data[group_name] = {}

        old_value = self.config_data[group_name].get(param_name, "N/A")
        self.config_data[group_name][param_name] = value
        print(f"✓ Updated {group_name}.{param_name}: {old_value} → {value}")

    def update_group(self, group_name: str, parameters: Dict[str, Any]) -> None:
        """Update multiple parameters in a group"""
        if group_name not in self.config_data:
            self.config_data[group_name] = {}

        for param_name, value in parameters.items():
            old_value = self.config_data[group_name].get(param_name, "N/A")
            self.config_data[group_name][param_name] = value
            print(f"✓ Updated {group_name}.{param_name}: {old_value} → {value}")

    def get_all_groups(self) -> Dict[str, Dict[str, Any]]:
        """Get all configuration groups"""
        return self.config_data

    def print_config(self, group_name: Optional[str] = None) -> None:
        """Print configuration (all or specific group)"""
        if group_name:
            if group_name in self.config_data:
                print(f"\n=== {group_name.upper()} Configuration ===")
                for key, value in self.config_data[group_name].items():
                    print(f"{key}: {value}")
            else:
                print(f"⚠ Group '{group_name}' not found")
        else:
            print("\n=== FULL Configuration ===")
            for group, params in self.config_data.items():
                print(f"\n[{group.upper()}]")
                for key, value in params.items():
                    print(f"  {key}: {value}")

    def reload_config(self) -> None:
        """Reload configuration from file"""
        print("🔄 Reloading configuration...")
        self.load_config()

    def reset_parameter(self, group_name: str, param_name: str) -> None:
        """Reset a parameter to its original value from file"""
        self.reload_config()
        original_value = self.get_parameter(group_name, param_name)
        print(f"✓ Reset {group_name}.{param_name} to original value: {original_value}")

    def backup_config(self, backup_suffix: str = "_backup") -> None:
        """Create a backup of current configuration"""
        backup_file = self.config_file.with_suffix(f'{backup_suffix}.json')
        try:
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(self.config_data, f, indent=2, ensure_ascii=False)
            print(f"✓ Backup created: {backup_file}")
        except Exception as e:
            print(f"✗ Error creating backup: {e}")

    def _resolve_paths(self) -> None:
        """Resolve path variables and create absolute paths"""
        if "file_paths" not in self.config_data:
            return

        paths = self.config_data["file_paths"].copy()
        resolved_paths = {}

        # First pass: resolve __FILE_DIR__ and simple paths
        for key, value in paths.items():
            if isinstance(value, str):
                if value == "__FILE_DIR__":
                    resolved_paths[key] = str(self.base_dir)
                elif not value.startswith("{"):
                    # Simple path, make it absolute relative to base_dir
                    resolved_paths[key] = str(self.base_dir / value)
                else:
                    resolved_paths[key] = value

        # Second pass: resolve path variables (multiple passes for nested references)
        max_iterations = 10
        for _ in range(max_iterations):
            changed = False
            for key, value in resolved_paths.items():
                if isinstance(value, str) and "{" in value:
                    # Replace variables with their resolved values
                    new_value = value
                    for var_key, var_value in resolved_paths.items():
                        if not "{" in var_value:  # Only use resolved values
                            new_value = new_value.replace(f"{{{var_key}}}", var_value)

                    if new_value != value:
                        resolved_paths[key] = new_value
                        changed = True

            if not changed:
                break

        # Final pass: convert to absolute paths and normalize
        for key, value in resolved_paths.items():
            if isinstance(value, str) and not "{" in value:
                path = Path(value)
                if not path.is_absolute():
                    path = self.base_dir / path
                resolved_paths[key] = str(path.resolve())

        # Update the config data
        self.config_data["file_paths"] = resolved_paths

    def get_path(self, path_name: str) -> str:
        """Get a resolved path by name"""
        return self.get_parameter("file_paths", path_name, "")

    def get_all_paths(self) -> Dict[str, str]:
        """Get all resolved paths"""
        return self.get_group("file_paths")

    def create_directories(self, path_names: Optional[list] = None) -> None:
        """Create directories for specified paths (or all paths if None)"""
        paths = self.get_all_paths()

        if path_names is None:
            path_names = list(paths.keys())

        for path_name in path_names:
            if path_name in paths:
                path = Path(paths[path_name])
                try:
                    path.mkdir(parents=True, exist_ok=True)
                    print(f"✓ Created directory: {path}")
                except Exception as e:
                    print(f"✗ Error creating directory {path}: {e}")

    def validate_paths(self) -> Dict[str, bool]:
        """Validate that all paths exist and are accessible"""
        paths = self.get_all_paths()
        validation_results = {}

        for path_name, path_str in paths.items():
            path = Path(path_str)
            try:
                if path.exists():
                    if path.is_dir():
                        validation_results[path_name] = True
                        print(f"✓ {path_name}: {path} (exists)")
                    else:
                        validation_results[path_name] = False
                        print(f"⚠ {path_name}: {path} (exists but not a directory)")
                else:
                    validation_results[path_name] = False
                    print(f"✗ {path_name}: {path} (does not exist)")
            except Exception as e:
                validation_results[path_name] = False
                print(f"✗ {path_name}: {path} (error: {e})")

        return validation_results


# Example usage and testing
if __name__ == "__main__":
    # Initialize config manager
    config = ConfigManager("trading_config.json")

    # Display current configuration
    config.print_config()

    # # Read parameters from different groups
    # print("\n=== Reading Parameters ===")
    # db_host = config.get_parameter("database", "host")
    # api_timeout = config.get_parameter("api", "timeout")
    # log_level = config.get_parameter("logging", "level")
    #
    # print(f"Database Host: {db_host}")
    # print(f"API Timeout: {api_timeout}")
    # print(f"Log Level: {log_level}")
    #
    # # Get entire group
    # print("\n=== Database Group ===")
    # db_config = config.get_group("database")
    # for key, value in db_config.items():
    #     print(f"{key}: {value}")

    # # Change parameters during runtime
    # print("\n=== Changing Parameters ===")
    # config.set_parameter("database", "port", 3306)
    # config.set_parameter("api", "timeout", 15)
    # config.set_parameter("logging", "level", "DEBUG")
    #
    # # Update multiple parameters at once
    # config.update_group("features", {
    #     "debug_mode": True,
    #     "maintenance_mode": True,
    #     "enable_caching": False
    # })
    #
    # # Display updated configuration
    # config.print_config("database")
    # config.print_config("features")
    #
    # # Save changes to file
    # config.save_config()
    #
    # # Create backup
    # config.backup_config()