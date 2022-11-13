import "./LoginPage.css";
import firebase from "./FirebaseBusiness";
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import Cookies from "js-cookie";

export default function LoginPage() {
  const navigate = useNavigate();
  const [email, setEmail] = useState();
  const [password, setPassword] = useState();
  const ref = firebase.firestore().collection("users");

  function login() {
    ref.onSnapshot((querySnapshot) => {
      var correctPassword = "";
      querySnapshot.forEach((doc) => {
        // console.log(doc.data().email);
        if (doc.data().email === email) {
          correctPassword = doc.data().password;
        }
      });
      if (password === correctPassword) {
        console.log("Login successful");
        Cookies.set("loggedInUser", email, { expires: 7 }); // 7 days
        navigate("/");
        console.log(Cookies.get("loggedInUser"));
      } else {
        console.log("no entry");
      }
    });
  }

  return (
    <div className="LoginPage">
      <div className="container">
        <h1 className="header">Login</h1>
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
        <button className="btn" onClick={login}>
          Login
        </button>
        <button className="btn partial-color">
          <a href="/signup">Sign Up</a>
        </button>
      </div>
    </div>
  );
}
