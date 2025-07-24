import { useState } from 'react';
import axios from 'axios';
import './SearchPage.css';
import { Link } from 'react-router-dom';

const SearchPage = () => {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);

  const searchProducts = async () => {
    try {
      const res = await axios.get(`http://localhost:8080/search?q=${query}`);
      setResults(res.data);
    } catch {
      alert('Search failed');
    }
  };

  return (
    <div className='search-container'>
      <div className='search-header'>
        <input
          className='search-input'
          placeholder='Search for products...'
          value={query}
          onChange={e => setQuery(e.target.value)}
        />
        <button className='search-button' onClick={searchProducts}>
          Search
        </button>
      </div>

      <div className='search-results'>
        {results.map(product => (
          <Link
            to={`/product/${product.id}`}
            key={product.id}
            style={{ textDecoration: 'none' }}
          >
            <div className='result-card'>
              <h4>{product.name}</h4>
              <p>Category: {product.category}</p>
              <p>Rating: {product.rating} ⭐</p>
            </div>
          </Link>
        ))}
      </div>
    </div>
  );
};

export default SearchPage;
