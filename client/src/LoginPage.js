import './LoginPage.css';
import firebase from './FirebaseBusiness'

export default function LoginPage() {
  const ref = firebase.firestore().collection("newcollection")
  function dw() {
    console.log("eeezzzzzzzzzzzzzzzzzzzzzzz")
    ref.add({ newjson: "randomstuff"})
  }
  return (
    <div className="LoginPage">
        <div className="container">
            <label>Email</label>
            <input className='input' type="text" />
            <label>Password</label>
            <input className='input' type="text" />
            <div className="btnContainer">
            <button className="btn" onClick={dw()}>Login</button>
            <button className="btn"><a href="/signup">Sign Up</a></button>
            </div>
        </div>
    </div>
  );
}