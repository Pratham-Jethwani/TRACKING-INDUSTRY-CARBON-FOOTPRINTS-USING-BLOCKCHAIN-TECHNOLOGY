pragma solidity ^0.8.20;

contract CarbonCredits {
    uint256 public constant DEFAULT_CREDITS = 10000;
    address public owner;

    // Updated Transaction struct with a flag indicating if it's incoming.
    struct Transaction {
        uint256 amount;
        address counterparty; // For emissions, this is address(0); for transfers, it's the other party.
        bool isIncoming;      // true = incoming; false = outgoing.
        uint256 timestamp;
    }

    mapping(address => uint256) private balances;
    mapping(address => Transaction[]) private transactionLogs;

    event AccountInitialized(address indexed user, uint256 amount, uint256 timestamp);
    event EmissionLogged(address indexed user, uint256 amount, uint256 timestamp);
    event CreditTransferred(address indexed from, address indexed to, uint256 amount, uint256 timestamp);

    constructor() {
        owner = msg.sender;
    }

    // For testing: resets the caller's balance to DEFAULT_CREDITS (500)
    function initializeAccount() public {
        balances[msg.sender] = DEFAULT_CREDITS;
        emit AccountInitialized(msg.sender, DEFAULT_CREDITS, block.timestamp);
    }

    // Log an emission (deduct credits and record an outgoing transaction)
    function logCarbonCredit(uint256 amount) public {
        require(amount <= balances[msg.sender], "Insufficient credits");
        balances[msg.sender] -= amount;
        transactionLogs[msg.sender].push(Transaction(amount, address(0), false, block.timestamp));
        emit EmissionLogged(msg.sender, amount, block.timestamp);
    }

    // Transfer credits: sender logs outgoing; receiver logs incoming.
    function transferCarbonCredit(address to, uint256 amount) public {
        require(to != address(0), "Invalid recipient address");
        require(amount <= balances[msg.sender], "Insufficient credits");

        balances[msg.sender] -= amount;
        balances[to] += amount;
        transactionLogs[msg.sender].push(Transaction(amount, to, false, block.timestamp));
        transactionLogs[to].push(Transaction(amount, msg.sender, true, block.timestamp));
        emit CreditTransferred(msg.sender, to, amount, block.timestamp);
    }

    // Returns the remaining credits of a user.
    function getRemainingCredits(address user) public view returns (uint256) {
        return balances[user];
    }

    // Instead of returning the entire array (which may cause ABI decoding issues), we return the count…
    function getTransactionCount(address user) public view returns (uint256) {
        return transactionLogs[user].length;
    }

    // …and a function to return an individual transaction.
    function getTransaction(address user, uint256 index) public view returns (Transaction memory) {
        require(index < transactionLogs[user].length, "Index out of bounds");
        return transactionLogs[user][index];
    }
}
