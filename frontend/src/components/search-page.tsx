import { FlexItem, Skeleton, Title } from "@patternfly/react-core";
import { GalleryView } from "./Gallery";
import { FakerProducts } from "./faker-products";
import { fetchSearch } from "../services/products";
import { useQuery } from "@tanstack/react-query";

export function SearchPage() {
  const { data, error, isLoading } = useQuery({
    queryKey: ["catalog"], // A unique key for this query
    queryFn: fetchSearch, // The async function to fetch data
  });

  const products = data ? data : FakerProducts();

  return (
    <FlexItem>
      <Title headingLevel={"h1"} style={{ paddingBottom: 20 }}>Search</Title>
      {isLoading ? (
        <Skeleton style={{ height: 200 }} />
      ) : (
        <GalleryView products={products} />
      )}
    </FlexItem>
  );
}
