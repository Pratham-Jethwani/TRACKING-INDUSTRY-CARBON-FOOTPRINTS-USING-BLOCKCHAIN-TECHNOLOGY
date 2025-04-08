import React, { useContext } from "react";
import { WalletContext } from "../contexts/WalletContext";

const InitializeAccount = () => {
  const { contract } = useContext(WalletContext);

  const handleInitialize = async () => {
    try {
      const tx = await contract.initializeAccount();
      await tx.wait();
      alert("Account initialized with 10000 credits!");
      window.location.reload();
    } catch (error) {
      console.error("Account initialization failed:", error);
      alert("Account initialization failed. Please try again.");
    }
  };

  return (
    <div style={{ marginBottom: "20px" }}>
      <button
        onClick={handleInitialize}
        style={{
          padding: "10px 20px",
          backgroundColor: "#6b46c1",
          color: "white",
          border: "none",
          borderRadius: "5px"
        }}
      >
        Initialize Account (Set 10000 Credits)
      </button>
    </div>
  );
};

export default InitializeAccount;
