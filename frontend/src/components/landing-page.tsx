import { FlexItem, Skeleton, Title } from "@patternfly/react-core";
import { fetchRecommendations } from "../services/products";
import { useQuery } from "@tanstack/react-query";
import { GalleryView } from "./Gallery";

export function LandingPage() {
  const { data, isError, isLoading } = useQuery({
    queryKey: ["recommnedations"], // A unique key for this query
    queryFn: fetchRecommendations, // The async function to fetch data
  });

  return (
      <FlexItem>
        <Title headingLevel={"h1"} style={{ paddingBottom: 20 }}>Product Recommendations</Title>
        {isLoading ? (
          <Skeleton style={{ height: 200 }} />
        ) : (
          isError ? <div>Error fetching recommendations</div> : <GalleryView products={data ?? []} />
        )}
      </FlexItem>
  );
}
