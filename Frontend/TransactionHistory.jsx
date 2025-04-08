import React, { useEffect, useState, useContext } from "react";
import { WalletContext } from "../contexts/WalletContext";

const TransactionHistory = () => {
  const { contract, walletAddress } = useContext(WalletContext);
  const [transactions, setTransactions] = useState([]);

  useEffect(() => {
    const fetchTransactions = async () => {
      if (contract && walletAddress) {
        try {
          const countBN = await contract.getTransactionCount(walletAddress);
          const count = countBN.toNumber();
          console.log("Transaction count:", count);
          const txs = [];
          for (let i = 0; i < count; i++) {
            const tx = await contract.getTransaction(walletAddress, i);
            txs.push(tx);
          }
          console.log("Fetched transactions:", txs);
          setTransactions(txs);
        } catch (error) {
          console.error("Error fetching transactions:", error);
          alert("Error fetching transaction history. Please ensure your contract is updated and try again.");
          setTransactions([]);
        }
      }
    };
    fetchTransactions();
  }, [contract, walletAddress]);

  return (
    <div style={{ marginBottom: "20px" }}>
      <h2>Transaction History</h2>
      {transactions.length === 0 ? (
        <p>No transactions yet.</p>
      ) : (
        <table style={{ width: "100%", borderCollapse: "collapse" }}>
          <thead>
            <tr>
              <th style={{ border: "1px solid black", padding: "8px" }}>Type</th>
              <th style={{ border: "1px solid black", padding: "8px" }}>Counterparty</th>
              <th style={{ border: "1px solid black", padding: "8px" }}>Amount</th>
              <th style={{ border: "1px solid black", padding: "8px" }}>Date</th>
            </tr>
          </thead>
          <tbody>
            {transactions.map((tx, index) => (
              <tr key={index}>
                <td style={{ border: "1px solid black", padding: "8px" }}>
                  {tx.isIncoming ? "Incoming" : "Outgoing"}
                </td>
                <td style={{ border: "1px solid black", padding: "8px" }}>
                  {tx.counterparty === "0x0000000000000000000000000000000000000000" ? "-" : tx.counterparty}
                </td>
                <td style={{ border: "1px solid black", padding: "8px" }}>
                  {tx.amount.toString()}
                </td>
                <td style={{ border: "1px solid black", padding: "8px" }}>
                  {new Date(tx.timestamp.toNumber() * 1000).toLocaleString()}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
};

export default TransactionHistory;
