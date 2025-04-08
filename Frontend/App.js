import React, { useContext } from "react";
import { WalletContext } from "./contexts/WalletContext";
import Dashboard from "./components/Dashboard";
import InitializeAccount from "./components/InitializeAccount";
import EmissionsLog from "./components/EmissionsLog";
import TransferForm from "./components/TransferForm";
import TransactionHistory from "./components/TransactionHistory";

function App() {
  const { walletAddress, connectWallet } = useContext(WalletContext);

  return (
    <div style={{ padding: "20px", fontFamily: "Arial, sans-serif" }}>
      <h1>Carbon Credits Dashboard</h1>
      {!walletAddress ? (
        <button onClick={connectWallet} style={{ padding: "10px 20px", fontSize: "16px" }}>
          Connect Wallet (MetaMask)
        </button>
      ) : (
        <>
          <Dashboard />
          <InitializeAccount />
          <EmissionsLog />
          <TransferForm />
          <TransactionHistory />
        </>
      )}
    </div>
  );
}

export default App;
