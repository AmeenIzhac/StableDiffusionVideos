import './SignUpPage.css';

export default function SignUpPage() {
  return (
    <div className="SignUpPage">
        <div className="container">
            <label>Name</label>
            <input className='input' type="text" />
            <label>Email</label>
            <input className='input' type="text" />
            <label>Password</label>
            <input className='input' type="password" />
            <div className="btnContainer">
            <button className="btn">Sign Up</button>
            <button className="btn"><a href="/login">Login</a></button>
            </div>
        </div>
    </div>
  );
}