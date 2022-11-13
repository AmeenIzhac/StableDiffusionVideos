import './Navbar.css'

export default function Navbar() {
    return (
        <div className="Navbar">
            <nav className='navbar'>
                <label className='logo'>AI VidGen</label>
                <ul>
                    <li className="pageLink"><a className="link" href='login'>Login</a></li>
                    <li className="pageLink"><a className="link" href='signup'>Sign Up</a></li>
                </ul>
            </nav>
        </div>
    );
  }

  