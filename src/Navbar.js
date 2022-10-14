import './Navbar.css'

export default function Navbar({isSuper, link, href}) {
    return (
        <div className="Navbar">
            <nav className='navbar'>
                <label className='logo'>AI VidGen {isSuper ? <p className="superTag">super</p> : null} </label>

                <li className="pageLink"><a className="link" href={href}>{link}</a></li>
            </nav>
        </div>
    );
  }

  