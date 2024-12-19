import json
import logging

from pathlib import Path
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

class ABI_Registry:
    """Centralized ABI registry with validation and signature mapping."""

    REQUIRED_METHODS = {
        'erc20': {'transfer', 'approve', 'transferFrom', 'balanceOf'},
        'uniswap': {'swapExactTokensForTokens', 'swapTokensForExactTokens', 'addLiquidity'},
        'sushiswap': {'swapExactTokensForTokens', 'swapTokensForExactTokens', 'addLiquidity'},
        'pancakeswap': {'swapExactTokensForTokens', 'swapTokensForExactTokens', 'addLiquidity'},
        'balancer': {'swap', 'addLiquidity'},
        'aave_flashloan': {'fn_RequestFlashLoan', 'executeOperation', 'ADDRESSES_PROVIDER', 'POOL'},
        'aave_lending': {'upgradeTo', 'implementation', 'initialize', 'admin'}
    }

    def __init__(self):
        self.abis: Dict[str, List[Dict]] = {}
        self.signatures: Dict[str, Dict[str, str]] = {}
        self.method_selectors: Dict[str, Dict[str, str]] = {}
        self._load_all_abis()

    def _load_all_abis(self) -> None:
        """Load and validate all ABIs at initialization."""
        try:
            abi_dir = Path(__file__).parent.parent / 'abi'
            
            # Centralize all ABI loading here
            abi_files = {
                'erc20': 'erc20_abi.json',
                'uniswap': 'uniswap_router_abi.json',
                'sushiswap': 'sushiswap_router_abi.json',
                'pancakeswap': 'pancakeswap_router_abi.json',
                'balancer': 'balancer_router_abi.json',
                'aave_flashloan': 'aave_flashloan_abi.json',
                'aave_lending': 'aave_lending_pool_abi.json'
            }

            for abi_type, filename in abi_files.items():
                try:
                    with open(abi_dir / filename, 'r') as f:
                        abi = json.load(f)
                    if self._validate_abi(abi, abi_type):
                        self.abis[abi_type] = abi
                        self._extract_signatures(abi, abi_type)
                        logger.debug(f"Loaded and validated {abi_type} ABI")
                except Exception as e:
                    logger.error(f"Failed to load {abi_type} ABI: {e}")
                    raise

        except Exception as e:
            logger.error(f"Error in ABI loading: {e}")
            raise

    def _validate_abi(self, abi: List[Dict], abi_type: str) -> bool:
        """Validate ABI structure and required methods."""
        if not isinstance(abi, list):
            logger.error(f"Invalid ABI format for {abi_type}")
            return False

        found_methods = {
            item['name'] for item in abi 
            if item.get('type') == 'function' and 'name' in item
        }
        
        required = self.REQUIRED_METHODS.get(abi_type, set())
        if abi_type in ['aave_flashloan', 'aave_lending']:
            # For Aave contracts, require at least one of the required methods
            if not (required & found_methods):  # Use intersection instead of subset
                missing = required - found_methods
                logger.error(f"No required methods found in {abi_type} ABI from: {missing}")
                return False
        else:
            # For other contracts, require all methods
            if not required.issubset(found_methods):
                missing = required - found_methods
                logger.error(f"Missing required methods in {abi_type} ABI: {missing}")
                return False

        return True

    def _extract_signatures(self, abi: List[Dict], abi_type: str) -> None:
        """Extract function signatures and method selectors."""
        from eth_abi.codec import ABICodec
        from eth_utils import function_signature_to_4byte_selector
        
        signatures = {}
        selectors = {}
        
        for item in abi:
            if item.get('type') == 'function':
                name = item.get('name')
                if name:
                    # Create function signature
                    inputs = ','.join(inp['type'] for inp in item.get('inputs', []))
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

