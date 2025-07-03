import { FlexItem, Skeleton, Title } from "@patternfly/react-core";
import { FakerProducts } from "./faker-products";
import { fetchRecommendations } from "../services/products";
import { useQuery } from "@tanstack/react-query";
import { GalleryView } from "./Gallery";

export function LandingPage() {
  const { data: recommendationItems, error: recommendationError, isLoading: isRecommendationsLoading } = useQuery({
    queryKey: ["recommnedations"], // A unique key for this query
    queryFn: fetchRecommendations, // The async function to fetch data
  });

  const products = recommendationItems ? recommendationItems : FakerProducts();

  return (
    <>
      <FlexItem>
        <Title headingLevel={"h1"} style={{ paddingBottom: 20 }}>Product Recommendations</Title>
        {isRecommendationsLoading ? (
          <Skeleton style={{ height: 200 }} />
        ) : (
          <GalleryView products={products} />
        )}
      </FlexItem>

    </>
  );
}
