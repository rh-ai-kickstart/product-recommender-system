import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { useState } from 'react';
import Preferences from './Preferences';
import Dashboard from './Dashboard';
import SearchPage from './SearchPage';
import ProductDetail from './ProductDetail';
import Login from './Login';
import NavBar from './NavBar';

function App() {
  const [user, setUser] = useState(null);
  const [preferences, setPreferences] = useState([]);

  // Define handleLogout inside App component
  const handleLogout = () => {
    setUser(null); // Clear user session
    setPreferences([]); // Clear onboarding state
  };

  if (!user) return <Login onLogin={setUser} />;
  if (preferences.length === 0)
    return <Preferences onContinue={setPreferences} />;

  return (
    <BrowserRouter>
      <NavBar user={user} onLogout={handleLogout} />
      <Routes>
        <Route path='/' element={<Dashboard user={user} />} />
        <Route path='/search' element={<SearchPage />} />
        <Route path='/product/:id' element={<ProductDetail />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
