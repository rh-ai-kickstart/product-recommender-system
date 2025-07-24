import { useState } from 'react';
import './Preferences.css';

const Preferences = ({ onContinue }) => {
  const categories = ['Electronics', 'Books', 'Clothing', 'Home', 'Sports'];
  const [selected, setSelected] = useState(null);

  const selectCategory = category => {
    setSelected(prev => (prev === category ? null : category));
  };

  return (
    <div className='preferences-container'>
      <h2>Select Your Main Interest</h2>
      <div>
        {categories.map(cat => (
          <button
            key={cat}
            onClick={() => selectCategory(cat)}
            className={`category-button ${selected === cat ? 'selected' : ''}`}
          >
            {cat}
          </button>
        ))}
      </div>
      <button
        className='continue-button'
        onClick={() => onContinue(selected)}
        disabled={!selected}
      >
        Continue
      </button>
    </div>
  );
};

export default Preferences;
