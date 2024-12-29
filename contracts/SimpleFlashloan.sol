// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "https://github.com/aave/aave-v3-core/blob/master/contracts/flashloan/base/FlashLoanSimpleReceiverBase.sol";
import "https://github.com/aave/aave-v3-core/blob/master/contracts/interfaces/IPoolAddressesProvider.sol";
import "https://github.com/aave/aave-v3-core/blob/master/contracts/dependencies/openzeppelin/contracts/IERC20.sol";

contract SimpleFlashLoan is FlashLoanSimpleReceiverBase {
    address payable public owner;

    event FlashLoanRequested(address token, uint256 amount);
    event FlashLoanExecuted(address token, uint256 amount, uint256 premium, bool success);

    constructor(address _addressProvider) FlashLoanSimpleReceiverBase(IPoolAddressesProvider(_addressProvider)) {
        owner = payable(msg.sender); // Contract deployer becomes the owner
    }

    // Modifier to allow only the owner to perform sensitive functions
    modifier onlyOwner() {
        require(msg.sender == owner, "Not contract owner");
        _;
    }

    // Request a flash loan from Aave's pool
    function fn_RequestFlashLoan(address _token, uint256 _amount) public onlyOwner {
        address receiverAddress = address(this);
        address asset = _token;
        uint256 amount = _amount;
        bytes memory params = "";  // Empty params can be expanded for more complex logic
        uint16 referralCode = 0;   // Optional referral code

        // Emit an event before triggering the flash loan
        emit FlashLoanRequested(_token, _amount);
        
        // Execute the flash loan
        POOL.flashLoanSimple(
            receiverAddress,
            asset,
            amount,
            params,
            referralCode
        );
    }

    // This function is called by Aave after receiving the loaned amount
    function executeOperation(
        address asset,
        uint256 amount,
        uint256 premium,
        address initiator,
        bytes calldata params
    ) external override returns (bool) {
        // Custom logic for MEV bot, arbitrage, liquidation, etc., should be placed here
        // Flash loan execution logic handled outside the contract in Python

        // Approve Aave to withdraw the loan + premium
        uint256 totalAmount = amount + premium;
        require(IERC20(asset).approve(address(POOL), totalAmount), "Approval failed");

        // Emit event after execution for monitoring/logging
        emit FlashLoanExecuted(asset, amount, premium, true);

        return true;
    }

    // Allow the owner to withdraw any ERC20 tokens stuck in the contract
    function withdrawToken(address _tokenAddress) public onlyOwner {
        IERC20 token = IERC20(_tokenAddress);
        uint256 balancer_abi = token.balanceOf(address(this));
        require(balancer_abi > 0, "No tokens to withdraw");
        token.transfer(owner, balancer_abi);
    }

    // Function to withdraw ETH if the contract receives any
    function withdrawETH() public onlyOwner {
        uint256 balancer_abi = address(this).balancer_abi;
        require(balancer_abi > 0, "No ETH to withdraw");
        owner.transfer(balancer_abi);
    }

    // Fallback function to receive ETH
    receive() external payable {}
}