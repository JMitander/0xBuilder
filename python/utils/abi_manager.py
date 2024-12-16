import json
import logging
from typing import Dict, List, Optional, Any
import aiofiles
from pathlib import Path

logger = logging.getLogger(__name__)

class ABI_Manager:
    """Centralized ABI management system."""

    def __init__(self):
        self.cached_abis: Dict[str, List[Dict[str, Any]]] = {}
        self.abi_paths = {
            'uniswap': 'uniswap_router_abi.json',
            'sushiswap': 'sushiswap_router_abi.json',
            'pancakeswap': 'pancakeswap_router_abi.json',
            'balancer': 'balancer_router_abi.json',
            'erc20': 'erc20_abi.json'
        }

    async def load_abi(self, abi_name: str) -> Optional[List[Dict[str, Any]]]:
        """Load ABI with validation and caching."""
        if abi_name in self.cached_abis:
            return self.cached_abis[abi_name]

        try:
            base_path = Path(__file__).parent.parent.parent / 'abi'
            abi_path = base_path / self.abi_paths.get(abi_name, f"{abi_name}_abi.json")
            
            if not abi_path.exists():
                logger.error(f"ABI file not found: {abi_path}")
                return None

            async with aiofiles.open(abi_path, 'r') as f:
                content = await f.read()
                abi = json.loads(content)

            if not isinstance(abi, list):
                logger.error(f"Invalid ABI format in {abi_path}")
                return None

            self.cached_abis[abi_name] = abi
            logger.debug(f"Successfully loaded {abi_name} ABI")
            return abi

        except Exception as e:
            logger.error(f"Error loading {abi_name} ABI: {e}")
            return None

    def validate_abi(self, abi: List[Dict[str, Any]], required_methods: List[str]) -> bool:
        """Validate ABI contains required methods."""
        try:
            found_methods = {func['name'] for func in abi if 'name' in func}
            missing = set(required_methods) - found_methods
            if missing:
                logger.error(f"ABI missing required methods: {missing}")
                return False
            return True
        except Exception as e:
            logger.error(f"ABI validation error: {e}")
            return False
