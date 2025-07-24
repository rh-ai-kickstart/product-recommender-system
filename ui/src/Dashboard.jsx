import { useEffect, useState } from 'react';
import axios from 'axios';
import './Dashboard.css';
import { Link } from 'react-router-dom';

const Dashboard = ({ user }) => {
  const [recommendations, setRecommendations] = useState([]);

  useEffect(() => {
    axios
      .get(`http://localhost:8080/recommendations?userId=${user.id}`)
      .then(res => setRecommendations(res.data))
      .catch(() => alert('Failed to load recommendations'));
  }, [user]);

  return (
    <div className='dashboard-container'>
      <h2>Recommended for {user.name}</h2>
      <div className='recommendation-grid'>
        {recommendations.map(prod => (
          <Link
            to={`/product/${prod.id}`}
            key={prod.id}
            style={{ textDecoration: 'none' }}
          >
            <div className='product-card'>
              <h4>{prod.name}</h4>
              <p>Category: {prod.category}</p>
              <p>Rating: {prod.rating} ⭐</p>
            </div>
          </Link>
        ))}
      </div>
    </div>
  );
};

export default Dashboard;
