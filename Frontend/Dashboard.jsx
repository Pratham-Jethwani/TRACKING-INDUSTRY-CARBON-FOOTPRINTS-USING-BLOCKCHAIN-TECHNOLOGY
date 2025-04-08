import React, { useContext, useEffect, useState } from "react";
import { WalletContext } from "../contexts/WalletContext";

const Dashboard = () => {
  const { walletAddress, contract } = useContext(WalletContext);
  const [balance, setBalance] = useState(null);

  useEffect(() => {
    const fetchBalance = async () => {
      if (contract && walletAddress) {
        try {
          const bal = await contract.getRemainingCredits(walletAddress);
          setBalance(bal.toString());
        } catch (error) {
          console.error("Error fetching balance:", error);
        }
      }
    };
    fetchBalance();
  }, [contract, walletAddress]);

  return (
    <div style={{ marginBottom: "20px" }}>
      <h2>Dashboard</h2>
      <p>
        <strong>Wallet Address:</strong> {walletAddress}
      </p>
      <p>
        <strong>Remaining Credits:</strong>{" "}
        {balance !== null ? balance : "Loading..."}
      </p>
    </div>
  );
};

export default Dashboard;
