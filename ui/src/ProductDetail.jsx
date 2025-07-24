import { useParams } from 'react-router-dom';
import { useEffect, useState } from 'react';
import axios from 'axios';
import './ProductDetail.css';

const ProductDetail = () => {
  const { id } = useParams();
  const [product, setProduct] = useState(null);

  useEffect(() => {
    axios
      .get(`http://localhost:8080/product/${id}`)
      .then(res => setProduct(res.data));
  }, [id]);

  return product ? (
    <div className='product-detail-container'>
      <div className='product-detail-box'>
        <h2>{product.name}</h2>
        <span className='category'>{product.category}</span>
        <p className='rating'>Rating: {product.rating} ⭐</p>
        <p className='description'>{product.description}</p>
      </div>
    </div>
  ) : (
    <p style={{ padding: '2rem', textAlign: 'center' }}>Loading...</p>
  );
};

export default ProductDetail;
