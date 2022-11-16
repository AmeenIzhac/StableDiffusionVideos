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
                <label className='logo'>AI VidGen</label>
                { loggedIn ? 
                <ul>
                    <li className="pageLink link"><a className="link" href='gallery'>Gallery</a></li>
                    <li className="pageLink" >
                        <button className='pageLink link'onClick={logout}>Logout</button>
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

  