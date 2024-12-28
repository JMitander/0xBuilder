# /home/mitander/0xBuilder/abi_registry.py
import json
import logging

from pathlib import Path
from typing import Dict, List, Optional, Set, Any

from utils import function_signature_to_4byte_selector

import aiofiles

logger = logging.getLogger("ABI_Registry")

class ABI_Registry:
    """Centralized ABI registry with validation and signature mapping."""

    REQUIRED_METHODS = {
        'erc20': {'transfer', 'approve', 'transferFrom', 'balanceOf'},
        'uniswap': {'swapExactTokensForTokens', 'swapTokensForExactTokens', 'addLiquidity', 'getAmountsOut'},
        'sushiswap': {'swapExactTokensForTokens', 'swapTokensForExactTokens', 'addLiquidity', 'getAmountsOut'},
        'pancakeswap': {'swapExactTokensForTokens', 'swapTokensForExactTokens', 'addLiquidity', 'getAmountsOut'},
        'balancer': {'swap', 'addLiquidity'},
        'aave_flashloan': {'fn_RequestFlashLoan', 'executeOperation', 'ADDRESSES_PROVIDER', 'POOL'},
        'aave': {'ADDRESSES_PROVIDER', 'getReservesList', 'getReserveData'}
    }

    def __init__(self):
        self.abis: Dict[str, List[Dict]] = {}
        self.signatures: Dict[str, Dict[str, str]] = {}
        self.method_selectors: Dict[str, Dict[str, str]] = {}
        self._initialized: bool = False

    async def initialize(self) -> None:
        """Async method to load and validate all ABIs at initialization."""
        if self._initialized:
            logger.debug("ABI_Registry already initialized.")
            return
        await self._load_all_abis()
        self._initialized = True
        logger.debug("ABI_Registry initialization complete.")

    async def _load_all_abis(self) -> None:
        """Load and validate all ABIs at initialization."""
        abi_dir = Path(__file__).parent.parent / 'abi'

        abi_files = {
            'erc20': 'erc20_abi.json',
            'uniswap': 'uniswap_abi.json',
            'sushiswap': 'sushiswap_router_abi.json',
            'pancakeswap': 'pancakeswap_router_abi.json',
            'balancer': 'balancer_router_abi.json',
            'aave_flashloan': 'aave_flashloan.abi.json',
            'aave': 'aave_pool_abi.json'
        }

        # Define critical ABIs that are essential for the application
        critical_abis = {'erc20', 'uniswap'}

        for abi_type, filename in abi_files.items():
            abi_path = abi_dir / filename
            try:
                abi = await self._load_abi_from_path(abi_path, abi_type)
                self.abis[abi_type] = abi
                self._extract_signatures(abi, abi_type)
                logger.debug(f"Loaded and validated {abi_type} ABI from {abi_path}")
            except FileNotFoundError:
                logger.error(f"ABI file not found for {abi_type}: {abi_path}")
                if abi_type in critical_abis:
                    raise
                else:
                    logger.warning(f"Skipping non-critical ABI: {abi_type}")
            except ValueError as ve:
                logger.error(f"Validation failed for {abi_type} ABI: {ve}")
                if abi_type in critical_abis:
                    raise
                else:
                    logger.warning(f"Skipping non-critical ABI: {abi_type}")
            except json.JSONDecodeError as je:
                logger.error(f"JSON decode error for {abi_type} ABI: {je}")
                if abi_type in critical_abis:
                    raise
                else:
                    logger.warning(f"Skipping non-critical ABI: {abi_type}")
            except Exception as e:
                logger.error(f"Unexpected error loading {abi_type} ABI: {e}")
                if abi_type in critical_abis:
                    raise
                else:
                    logger.warning(f"Skipping non-critical ABI: {abi_type}")

    async def _load_abi_from_path(self, abi_path: Path, abi_type: str) -> List[Dict]:
        """Loads and validates the ABI from a given path."""
        try:
            if not abi_path.exists():
                logger.error(f"ABI file not found: {abi_path}")
                raise FileNotFoundError(f"ABI file not found: {abi_path}")

            async with aiofiles.open(abi_path, 'r', encoding='utf-8') as f:
                abi_content = await f.read()
                abi = json.loads(abi_content)
                logger.debug(f"ABI content loaded from {abi_path}")

            if not self._validate_abi(abi, abi_type):
                raise ValueError(f"Validation failed for {abi_type} ABI from file {abi_path}")

            return abi
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error for {abi_type} in file {abi_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading ABI {abi_type}: {e}")
            raise

    async def load_abi(self, abi_type: str) -> Optional[List[Dict]]:
        """Load specific ABI type with validation."""
        try:
            abi_dir = Path(__file__).parent.parent / 'abi'
            abi_files = {
                'erc20': 'erc20_abi.json',
                'uniswap': 'uniswap_abi.json',
                'sushiswap': 'sushiswap_router_abi.json',
                'pancakeswap': 'pancakeswap_router_abi.json',
                'balancer': 'balancer_router_abi.json',
                'aave_flashloan': 'aave_flashloan.abi.json',
                'aave': 'aave_pool_abi.json'
            }
            
            if abi_type not in abi_files:
                logger.error(f"Unknown ABI type: {abi_type}")
                return None

            abi_path = abi_dir / abi_files[abi_type]
            abi = await self._load_abi_from_path(abi_path, abi_type)
            self.abis[abi_type] = abi
            self._extract_signatures(abi, abi_type)
            logger.debug(f"Loaded and validated {abi_type} ABI")
            return abi
        except Exception as e:
            logger.error(f"Error loading ABI {abi_type}: {e}")
            return None

    def _validate_abi(self, abi: List[Dict], abi_type: str) -> bool:
        """Validate ABI structure and required methods."""
        if not isinstance(abi, list):
            logger.error(f"Invalid ABI format for {abi_type}")
            return False

        found_methods = {
            item.get('name') for item in abi
            if item.get('type') == 'function' and 'name' in item
        }

        required = self.REQUIRED_METHODS.get(abi_type, set())
        if not required.issubset(found_methods):
            missing = required - found_methods
            logger.error(f"Missing required methods in {abi_type} ABI: {missing}")
            return False

        return True

    def _extract_signatures(self, abi: List[Dict], abi_type: str) -> None:
        """Extract function signatures and method selectors."""
        signatures = {}
        selectors = {}

        for item in abi:
            if item.get('type') == 'function':
                name = item.get('name')
                if name:
                    # Create function signature
                    inputs = ','.join(inp.get('type', '') for inp in item.get('inputs', []))
                    signature = f"{name}({inputs})"

                    # Generate selector
                    selector = function_signature_to_4byte_selector(signature)
                    hex_selector = selector.hex()

                    signatures[name] = signature
                    selectors[hex_selector] = name

        self.signatures[abi_type] = signatures
        self.method_selectors[abi_type] = selectors

    def get_abi(self, abi_type: str) -> Optional[List[Dict]]:
        """Get validated ABI by type."""
        return self.abis.get(abi_type)

    def get_method_selector(self, selector: str) -> Optional[str]:
        """Get method name from selector, checking all ABIs."""
        for abi_type, selectors in self.method_selectors.items():
            if selector in selectors:
                return selectors[selector]
        return None

    def get_function_signature(self, abi_type: str, method_name: str) -> Optional[str]:
        """Get function signature by ABI type and method name."""
        return self.signatures.get(abi_type, {}).get(method_name)