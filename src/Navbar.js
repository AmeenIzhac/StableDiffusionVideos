import './Navbar.css'

export default function Navbar({link, href}) {
    return (
        <div className="Navbar">
            <nav className="navbar">
                <label>AI VidGen</label>
                <li><a href={href}>{link}</a></li>
            </nav>
        </div>
    );
  }

  