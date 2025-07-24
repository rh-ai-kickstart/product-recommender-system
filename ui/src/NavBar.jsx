import { Link, useNavigate } from 'react-router-dom';
import { useState } from 'react';
import './NavBar.css';

const NavBar = ({ user, onLogout }) => {
  const navigate = useNavigate();
  const [menuOpen, setMenuOpen] = useState(false);

  const handleLogout = () => {
    onLogout();
    navigate('/');
  };

  const toggleMenu = () => setMenuOpen(prev => !prev);

  return (
    <nav className='navbar'>
      <div className='navbar-logo'>🧠 Recommender</div>

      <button className='hamburger' onClick={toggleMenu}>
        ☰
      </button>

      <div className={`navbar-links ${menuOpen ? 'open' : ''}`}>
        <Link to='/' onClick={() => setMenuOpen(false)}>
          Dashboard
        </Link>
        <Link to='/search' onClick={() => setMenuOpen(false)}>
          Search
        </Link>
        <Link to='/history' onClick={() => setMenuOpen(false)}>
          History
        </Link>
        <button onClick={handleLogout}>Logout</button>
      </div>
    </nav>
  );
};

export default NavBar;
