import Cookies from 'js-cookie';
import './Navbar.css'

export default function Navbar({loggedIn, setLoggedIn}) {
    const logout = () => {
        Cookies.remove("loggedInUser")
        setLoggedIn(false);
    }

    return (
        <div className="Navbar">
            <nav className='navbar'>
                <label className='logo'>SD VidGen</label>
                { loggedIn ? 
                <ul>
                    <li className="pageLink"><a className='link' href='gallery'>Gallery</a></li>
                    <li  className='pageLink' >
                        <button className='buttonLink' onClick={logout}>Logout</button>
                    </li>
                </ul>
                :
                <ul>
                    <li className="pageLink link"><a className="link" href='login'>Login</a></li>
                    <li className="pageLink link"><a className="link" href='signup'>Sign Up</a></li>
                </ul>
                }
            </nav>
        </div>
    );
  }

  