#!/usr/bin/env python3
"""
organize_0xbuilder.py

A script to automatically organize the 0xBuilder MEV Bot project directory.
It moves Python and JavaScript files to their respective directories based on predefined rules,
excluding the 'abi' and 'Utils' directories with token addresses and symbols.

Usage:
    python organize_0xbuilder.py [--root /path/to/0xBuilder]

If no root directory is specified, the current working directory is used.
"""

import os
import shutil
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Organize 0xBuilder project directory.")
    parser.add_argument(
        '--root',
        type=str,
        default='.',
        help='Root directory of the 0xBuilder project (default: current directory)'
    )
    return parser.parse_args()

def is_abi_or_utils(file_path, root_path):
    """Check if the file is under 'abi/' or 'Utils/' directories that should be excluded."""
    relative_path = file_path.relative_to(root_path)
    if relative_path.parts[0] in ['abi', 'Utils']:
        # Further exclude specific files under Utils/
        if relative_path.parts[0] == 'Utils':
            if relative_path.name in ['token_addresses.json', 'token_symbols.json', 'erc20_signatures.json']:
                return True
        return True
    return False

def is_shared_file(file_path, root_path):
    """Check if the file should be moved to the 'Shared/' directory."""
    shared_files = {
        'DataClass.js',
        'Errors.js',
        'Logger.js',
        'Semaphore.js',
        'StrategyConfiguration.js',
        'StrategyExecutionError.js',
        'StrategyPerformanceMetrics.js',
        'loading_bar.js',
        'logger.js'
    }
    relative_path = file_path.relative_to(root_path)
    if relative_path.parts[0] == 'Utils' and relative_path.parts[1] == 'Javascript':
        if relative_path.name in shared_files:
            return True
    if relative_path.parts[0] == 'Logging':
        if relative_path.name in shared_files:
            return True
    return False

def get_target_path(file_path, root_path):
    """
    Determine the target path for the given file based on its current location and type.
    Returns the target directory path or None if the file should remain in place.
    """
    relative_path = file_path.relative_to(root_path)
    parts = relative_path.parts

    # Handle shared files
    if is_shared_file(file_path, root_path):
        return root_path / 'Shared' / file_path.name

    # Handle files under Analysis, Configuration, Engine for Python and JavaScript
    if len(parts) >= 3:
        category, language, filename = parts[0], parts[1], parts[2]
        if category in ['Analysis', 'Configuration', 'Engine']:
            if language.lower() == 'python':
                if category == 'Engine':
                    target_dir = root_path / 'Python' / 'Core'
                else:
                    target_dir = root_path / 'Python' / category
                return target_dir / filename
            elif language.lower() in ['javascript', 'js']:
                if category == 'Engine':
                    target_dir = root_path / 'JavaScript' / 'Core'
                else:
                    target_dir = root_path / 'JavaScript' / category
                return target_dir / filename

    # Handle Contracts/*.sol
    if len(parts) >= 2 and parts[0] == 'Contracts' and file_path.suffix == '.sol':
        target_python = root_path / 'Python' / 'Contracts' / file_path.name
        target_js = root_path / 'JavaScript' / 'Contracts' / file_path.name
        return [target_python, target_js]

    # Handle files directly under root
    if len(parts) == 1:
        filename = parts[0]
        if filename.endswith('.py') and filename != '0xBuilder.py':
            return root_path / 'Python' / filename
        if filename.endswith('.js') and filename != 'index.js':
            return root_path / 'JavaScript' / filename
        if filename == '0xBuilder.py':
            return root_path / 'Python' / filename
        if filename == 'index.js':
            return root_path / 'JavaScript' / filename
        if filename in ['requirements.txt', 'Package.json']:
            if filename == 'requirements.txt':
                return root_path / 'Python' / filename
            if filename == 'Package.json':
                return root_path / 'JavaScript' / 'package.json'
        if filename in ['LICENSE', 'CONTRIBUTING.md', 'README.md', '.env', '.env.', '.gitignore']:
            return root_path / filename  # Remain in root

    return None  # No action needed

def move_file(file_path, target_paths):
    """Move the file to the target path(s)."""
    if isinstance(target_paths, list):
        for target in target_paths:
            if not target.parent.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Created directory: {target.parent}")
            try:
                shutil.move(str(file_path), str(target))
                logger.debug(f"Moved {file_path} -> {target}")
            except Exception as e:
                logger.error(f"Failed to move {file_path} to {target}: {e}")
    else:
        target = target_paths
        if not target.parent.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {target.parent}")
        try:
            shutil.move(str(file_path), str(target))
            logger.debug(f"Moved {file_path} -> {target}")
        except Exception as e:
            logger.error(f"Failed to move {file_path} to {target}: {e}")

def organize_project(root_dir):
    """Organize the 0xBuilder project directory."""
    root_path = Path(root_dir).resolve()
    logger.debug(f"Starting organization in root directory: {root_path}")

    # Create Shared directory if it doesn't exist
    shared_dir = root_path / 'Shared'
    if not shared_dir.exists():
        shared_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created directory: {shared_dir}")

    # Traverse the directory tree
    for dirpath, dirnames, filenames in os.walk(root_path):
        current_dir = Path(dirpath)

        # Skip 'abi/' and 'Utils/' directories
        if 'abi' in current_dir.parts:
            continue
        if 'Utils' in current_dir.parts:
            continue

        for filename in filenames:
            file_path = current_dir / filename

            # Skip if file is under 'abi/' or 'Utils/' (already handled above)
            if is_abi_or_utils(file_path, root_path):
                continue

            # Determine target path
            target = get_target_path(file_path, root_path)
            if target:
                move_file(file_path, target)
    
    # Special handling for Contracts/*.sol to ensure they are in both Python and JavaScript
    contracts_dir = root_path / 'Contracts'
    python_contracts_dir = root_path / 'Python' / 'Contracts'
    js_contracts_dir = root_path / 'JavaScript' / 'Contracts'

    if contracts_dir.exists():
        for sol_file in contracts_dir.glob('*.sol'):
            # Move to Python/Contracts/
            if not python_contracts_dir.exists():
                python_contracts_dir.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Created directory: {python_contracts_dir}")
            try:
                shutil.copy2(str(sol_file), str(python_contracts_dir / sol_file.name))
                logger.debug(f"Copied {sol_file} -> {python_contracts_dir / sol_file.name}")
            except Exception as e:
                logger.error(f"Failed to copy {sol_file} to {python_contracts_dir}: {e}")

            # Move to JavaScript/Contracts/
            if not js_contracts_dir.exists():
                js_contracts_dir.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Created directory: {js_contracts_dir}")
            try:
                shutil.copy2(str(sol_file), str(js_contracts_dir / sol_file.name))
                logger.debug(f"Copied {sol_file} -> {js_contracts_dir / sol_file.name}")
            except Exception as e:
                logger.error(f"Failed to copy {sol_file} to {js_contracts_dir}: {e}")

    logger.debug("Organization complete.")

def main():
    args = parse_arguments()
    organize_project(args.root)

if __name__ == "__main__":
    main()
