import { PageSection, Title } from "@patternfly/react-core";
import { Carousel } from "./Carousel/carousel";
import { faker } from "@faker-js/faker";
import { GalleryView } from "./gallery";


interface ProductData {
  id: number;
  title: string;
  description: string;
  price: string;
  imageUrl: string;
  rating: string;
}

export function AccountPage() {
  function capitalizeFirstLetter(val: string) {
    return String(val).charAt(0).toUpperCase() + String(val).slice(1);
  }

  // Fetch Product Recommendations from backend
  function createRandomProducts() {
    const myProduct: ProductData = {
      id: faker.number.int(),
      title: capitalizeFirstLetter(
        faker.word.noun({ length: { min: 5, max: 7 } })
      ),
      description: faker.lorem.lines(1),
      price: faker.finance.amount({ min: 5, max: 20, dec: 2, symbol: "$" }),
      imageUrl: faker.image.urlLoremFlickr(),
      rating: faker.finance.amount({ min: 0, max: 5, dec: 2 }),
    };
    return myProduct;
  }
  const productsRecommended = faker.helpers.multiple(createRandomProducts, {
    count: 10,
  });
  const highlyRecProducts = faker.helpers.multiple(createRandomProducts, {
    count: 5,
  });
  const trendingProducts = faker.helpers.multiple(createRandomProducts, {
    count: 8,
  });

  return (
    <>
      <PageSection hasBodyWrapper={false}>
        <Title headingLevel={"h1"} style={{ marginTop: "15px" }}>
          Product Recommendations
        </Title>
      </PageSection>
      <Carousel products={productsRecommended} />
      <PageSection hasBodyWrapper={false}>
        <Title headingLevel={"h1"} style={{ marginTop: "15px" }}>
          Trending Products
        </Title>
        <GalleryView products={trendingProducts} />
      </PageSection>
      <PageSection hasBodyWrapper={false}>
        <Title headingLevel={"h1"} style={{ marginTop: "15px" }}>
          Highly Recommended
        </Title>
        <Carousel products={highlyRecProducts} />
      </PageSection>
    </>
  );
}
