import "./SignUpPage.css";
import "./LoginPage.css";
import firebase from "./FirebaseBusiness";
import { useState } from "react";
import { useNavigate } from "react-router-dom";

export default function SignUpPage() {
  const navigate = useNavigate();
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const ref = firebase.firestore().collection("users");
  function addUser() {
    ref.add({ name: name, email: email, password: password });
    setName("");
    setEmail("");
    setPassword("");
    navigate("/login");
  }

  return (
    <div className="SignUpPage">
      <h1><a href="/" className="back-to-search">&larr; Back To Video Generation</a></h1>
      <div className="container">
        <h1 className="header">Sign Up</h1>
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          className="input"
          type="text"
          placeholder="Name"
        />
        <input
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          className="input"
          type="text"
          placeholder="Email"
        />
        <input
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          className="input"
          type="password"
          placeholder="Password"
        />
        <button className="btn" onClick={addUser}>
          Sign Up
        </button>
        <button className="btn partial-color">
          <a href="/login">Login</a>
        </button>
      </div>
    </div>
  );
}
