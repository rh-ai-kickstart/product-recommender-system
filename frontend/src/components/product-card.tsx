import { Card, CardTitle, CardBody, CardFooter, CardHeader, Divider, Flex, FlexItem } from "@patternfly/react-core";
import { auto } from "@patternfly/react-core/dist/esm/helpers/Popper/thirdparty/popper-core";
import { StarIcon } from "@patternfly/react-icons";

interface ProductData {
  id: number;
  title: string;
  description: string;
  price: string;
  imageUrl: string;
  rating: string;
}

interface ProductCardProps {
  product: ProductData; // The 'product' prop will be of type ProductData
  index: number; // The 'index' prop will be a number
}

export const ProductCard: React.FunctionComponent<ProductCardProps> = ({ product, index }) => {
  const curCardCount = index + 1;
  const cardId = `featured-blog-post-${curCardCount}`;
  const actionId = `card-article-input-${curCardCount}`;
  const cardTitleId = `featured-blog-post-${curCardCount}-title`;

  return (
    <Card id={cardId} component="div" isClickable key={index} style={{height:400, overflowY: auto,}}>
      <CardHeader
        className="v6-featured-posts-card-header-img"
        selectableActions={{
          to: "#",
          selectableActionId: actionId,
          selectableActionAriaLabelledby: cardTitleId,
          name: 'homepage-card',
          isExternalLink: true,
        }}
        style={{
          backgroundImage: `url(${product.imageUrl})`,
          height: 200,
        }}
      ></CardHeader>
      <Divider />
      <CardTitle id={cardTitleId}>
        <Flex style={{ justifyContent: "space-between" }}>
          <FlexItem>{product.title}</FlexItem>
          <FlexItem>{product.rating} <StarIcon /></FlexItem>
        </Flex>
      </CardTitle>
      <CardBody style={{ color:"#707070" }}>
        {product.description}
      </CardBody>
      <CardFooter style={{ color:"#1F1F1F" }}>
        {product.price}
      </CardFooter>
    </Card>
  );
};
