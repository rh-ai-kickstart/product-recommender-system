import { useState, useEffect } from 'react';
import axios from 'axios';
import './Login.css';

const Login = ({ onLogin }) => {
  const [mode, setMode] = useState(''); // '' | 'login' | 'signup'
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [age, setAge] = useState('');
  const [gender, setGender] = useState('');
  const [preferences, setPreferences] = useState('');
  const [showPassword, setShowPassword] = useState(false); // 👁️ toggle

  useEffect(() => {
    // Reset all fields when switching mode
    setEmail('');
    setPassword('');
    setAge('');
    setGender('');
    setPreferences('');
    setShowPassword(false);
  }, [mode]);

  const handleSubmit = async () => {
    try {
      const url = `http://localhost:8080/${mode}`;
      const payload =
        mode === 'login'
          ? { email, password }
          : { email, password, age: parseInt(age), gender };

      const res = await axios.post(url, payload);
      onLogin(res.data.user);
    } catch (err) {
      alert(`${mode === 'login' ? 'Login' : 'Signup'} failed`);
    }
  };

  const getButtonClass = buttonType => {
    if (!mode)
      return buttonType === 'login' ? 'default-login' : 'default-signup';
    return mode === buttonType ? 'active' : 'inactive';
  };

  return (
    <div className='login-container'>
      <div className='login-box'>
        <div className='toggle-buttons'>
          <button
            onClick={() => setMode('login')}
            className={getButtonClass('login')}
          >
            Login
          </button>
          <button
            onClick={() => setMode('signup')}
            className={getButtonClass('signup')}
          >
            Sign Up
          </button>
        </div>

        {mode && (
          <>
            <h2>{mode === 'login' ? 'Login' : 'Sign Up'}</h2>
            <input
              className='login-input'
              placeholder='Email'
              value={email}
              onChange={e => setEmail(e.target.value)}
            />

            <div className='password-wrapper'>
              <input
                className='login-input'
                type={showPassword ? 'text' : 'password'}
                placeholder='Password'
                value={password}
                onChange={e => setPassword(e.target.value)}
              />
              <span
                className='toggle-password'
                onClick={() => setShowPassword(prev => !prev)}
                title={showPassword ? 'Hide password' : 'Show password'}
              >
                👁️
              </span>
            </div>

            {mode === 'signup' && (
              <>
                <input
                  className='login-input'
                  type='number'
                  placeholder='Age'
                  value={age}
                  onChange={e => setAge(e.target.value)}
                />
                <select
                  value={gender}
                  onChange={e => setGender(e.target.value)}
                  className='dropdown-input'
                >
                  <option value='' disabled>
                    Select Gender
                  </option>
                  <option value='M'>M</option>
                  <option value='F'>F</option>
                  <option value='Other'>Other</option>
                </select>
              </>
            )}

            <button onClick={handleSubmit}>
              {mode === 'login' ? 'Login' : 'Sign Up'}
            </button>
          </>
        )}
      </div>
    </div>
  );
};

export default Login;
