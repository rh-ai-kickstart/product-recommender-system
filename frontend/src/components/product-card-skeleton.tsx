import {
  Card,
  CardTitle,
  CardBody,
  CardFooter,
  CardHeader,
  Divider,
  Flex,
  FlexItem,
  Skeleton,
} from '@patternfly/react-core';

interface ProductCardSkeletonProps {
  index: number;
}

export const ProductCardSkeleton: React.FunctionComponent<
  ProductCardSkeletonProps
> = ({ index }) => {
  const curCardCount = index + 1;
  const cardId = `skeleton-card-${curCardCount}`;

  return (
    <Card
      id={cardId}
      component='div'
      style={{ height: 400, overflowY: 'hidden' }}
    >
      <CardHeader
        className='v6-featured-posts-card-header-img'
        style={{
          height: 200,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '10px',
        }}
      >
        <Skeleton
          style={{ borderRadius: '12px' }}
          width='100%'
          height='180px'
        />
      </CardHeader>
      <Divider />
      <CardTitle>
        <Flex style={{ justifyContent: 'space-between' }}>
          <FlexItem className='pf-v6-u-w-50'>
            <Skeleton width='100%' height='1.5rem' />
          </FlexItem>
          <FlexItem className='pf-v6-u-w-25'>
            <Skeleton width='100%' height='1.5rem' />
          </FlexItem>
        </Flex>
      </CardTitle>
      <CardBody style={{ color: '#707070' }}>
        <Skeleton
          width='100%'
          height='1rem'
          style={{ marginBottom: '0.5rem' }}
        />
        <Skeleton
          width='90%'
          height='1rem'
          style={{ marginBottom: '0.5rem' }}
        />
        <Skeleton width='70%' height='1rem' />
      </CardBody>
      <CardFooter style={{ color: '#1F1F1F' }}>
        <Skeleton width='30%' height='1.5rem' />
      </CardFooter>
    </Card>
  );
};
