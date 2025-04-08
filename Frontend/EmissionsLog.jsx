import React, { useState, useContext } from "react";
import { WalletContext } from "../contexts/WalletContext";

const EmissionsLog = () => {
  const { contract } = useContext(WalletContext);
  const [emissionAmount, setEmissionAmount] = useState("");

  const handleFetchPrediction = async () => {
    try {
      const response = await fetch("https://carboncredit-5lcz.onrender.com/fetch",{
        method: "POST",
      });

      const data = await response.json();
      if (data.prediction !== undefined) {
        const roundedPrediction = Math.round(data.prediction);
        setEmissionAmount(roundedPrediction);
        alert("Fetched predicted carbon credits: " + roundedPrediction);
      } else {
        alert("Failed to fetch prediction.");
      }
    } catch (error) {
      console.error("Error fetching prediction:", error);
      alert("Error contacting prediction API.");
    }
  };

  const handleLogEmission = async () => {
    if (!emissionAmount) {
      alert("Please fetch an emission amount before logging.");
      return;
    }
    try {
      const tx = await contract.logCarbonCredit(Math.round(emissionAmount));
      await tx.wait();
      alert("Emission logged successfully!");
      setEmissionAmount("");
      window.location.reload();
    } catch (error) {
      console.error("Emission logging failed:", error);
      if (error.message && error.message.includes("Insufficient credits")) {
        alert("Emission logging failed: Not sufficient balance to log the requested emission amount. Please check your balance.");
      } else {
        alert("Emission logging failed: " + error.message);
      }
    }
  };

  return (
    <div style={{ marginBottom: "20px" }}>
      <h2>Log Emission</h2>
      <input
        type="number"
        placeholder="Emission Amount (credits)"
        value={emissionAmount}
        disabled
        style={{ padding: "8px", marginRight: "10px", width: "200px", backgroundColor: "#f0f0f0" }}
      />
      <button
        onClick={handleFetchPrediction}
        style={{ padding: "10px 20px", marginRight: "10px", backgroundColor: "#4299e1", color: "white", border: "none" }}
      >
        Fetch
      </button>
      <button
        onClick={handleLogEmission}
        style={{ padding: "10px 20px", backgroundColor: "#48bb78", color: "white", border: "none" }}
      >
        Log Emission
      </button>
    </div>
  );
};

export default EmissionsLog;
