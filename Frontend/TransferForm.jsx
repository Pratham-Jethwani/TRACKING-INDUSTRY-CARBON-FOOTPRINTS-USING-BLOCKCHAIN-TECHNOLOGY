import React, { useState, useContext } from "react";
import { WalletContext } from "../contexts/WalletContext";

const TransferForm = () => {
  const { contract } = useContext(WalletContext);
  const [recipient, setRecipient] = useState("");
  const [amount, setAmount] = useState("");

  const handleTransfer = async () => {
    if (!recipient || !amount) {
      alert("Please fill in all fields.");
      return;
    }
    try {
      const tx = await contract.transferCarbonCredit(recipient, amount);
      await tx.wait();
      alert("Transfer successful!");
      setRecipient("");
      setAmount("");
      window.location.reload();
    } catch (error) {
      console.error("Transfer failed:", error);
      if (error.message && error.message.includes("Insufficient credits")) {
        alert("Transfer failed: Not sufficient balance for the transfer. Please check your balance.");
      } else {
        alert("Transfer failed: " + error.message);
      }
    }
  };

  return (
    <div style={{ marginBottom: "20px" }}>
      <h2>Transfer Credits</h2>
      <input
        type="text"
        placeholder="Recipient Address"
        value={recipient}
        onChange={(e) => setRecipient(e.target.value)}
        style={{ padding: "8px", marginRight: "10px", width: "300px" }}
      />
      <input
        type="number"
        placeholder="Amount"
        value={amount}
        onChange={(e) => setAmount(e.target.value)}
        style={{ padding: "8px", marginRight: "10px", width: "150px" }}
      />
      <button
        onClick={handleTransfer}
        style={{
          padding: "10px 20px",
          backgroundColor: "#3182ce",
          color: "white",
          border: "none",
          borderRadius: "5px"
        }}
      >
        Transfer
      </button>
    </div>
  );
};

export default TransferForm;
