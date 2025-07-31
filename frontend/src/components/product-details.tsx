import {
  Button,
  Card,
  CardBody,
  CardFooter,
  CardTitle,
  Flex,
  FlexItem,
  Skeleton,
} from '@patternfly/react-core';
import StarRatings from 'react-star-ratings';
import { useEffect } from 'react';
import { useProductActions } from '../hooks';
import { Route } from '../routes/_protected/product/$productId';

export const ProductDetails = () => {
  // loads productId from route /product/$productId
  const { productId } = Route.useLoaderData();

  // Use our composite hook for all product actions
  const { product, error, isLoading, addToCart, isAddingToCart, recordClick } =
    useProductActions(productId);

  // Record that user viewed this product when component mounts or productId changes
  useEffect(() => {
    if (product && !isLoading) {
      recordClick();
    }
  }, [product, isLoading, productId]); // Removed recordClick from deps to prevent infinite loop

  if (error || !product) {
    return <div>Error fetching product</div>;
  }

  // Mutation for buying product now
  const handleBuyNow = () => {
    alert(product.product_name);
  };

  return (
    <>
      {isLoading ? (
        <Skeleton style={{ flex: 1, minWidth: 0, height: '100%' }} />
      ) : (
        <>
          <FlexItem style={{ flex: 1, minWidth: 0, height: '100%' }}>
            <Card style={{ height: '100%' }}>
              <CardBody style={{ height: '100%', padding: 0 }}>
                <img
                  src={product.img_link}
                  alt={product.product_name}
                  style={{
                    width: '100%',
                    height: '100%',
                    objectFit: 'cover',
                    display: 'block',
                  }}
                />
              </CardBody>
            </Card>
          </FlexItem>
          <FlexItem style={{ flex: 1, minWidth: 0, height: '100%' }}>
            <Card isPlain style={{ height: '100%' }}>
              <CardTitle style={{ fontSize: '2rem', fontWeight: 'bold' }}>
                {product.product_name}
              </CardTitle>
              <CardBody>
                <Flex direction={{ default: 'column' }}>
                  <FlexItem>
                    <StarRatings
                      rating={product.rating}
                      starRatedColor='black'
                      numberOfStars={5}
                      name='rating'
                      starDimension='18px'
                      starSpacing='1px'
                    />{' '}
                    {product.rating}
                  </FlexItem>
                  <FlexItem headers='h1'>${product.actual_price}</FlexItem>
                  <FlexItem>{product.about_product}</FlexItem>
                </Flex>
              </CardBody>
              <CardFooter>
                <Flex>
                  <FlexItem>
                    <Button
                      variant='secondary'
                      onClick={() => addToCart(1)}
                      isLoading={isAddingToCart}
                      isDisabled={isAddingToCart}
                    >
                      {isAddingToCart ? 'Adding...' : 'Add to Cart'}
                    </Button>
                  </FlexItem>
                  <FlexItem>
                    <Button variant='primary' onClick={handleBuyNow}>
                      Buy Now
                    </Button>
                  </FlexItem>
                </Flex>
              </CardFooter>
            </Card>
          </FlexItem>
        </>
      )}
    </>
  );
};
