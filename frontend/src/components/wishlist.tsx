import { GalleryView } from "./Gallery";
import { Title, FlexItem, Skeleton } from "@patternfly/react-core";
import { fetchWishlist } from "../services/products";
import { useQuery } from "@tanstack/react-query";
import { FakerProducts } from "./faker-products";

export function Wishlist() {
  const { data, error, isLoading } = useQuery({
    queryKey: ["wishlist"], // A unique key for this query
    queryFn: fetchWishlist, // The async function to fetch data
  });
  const products = data ? data : FakerProducts();

  return (
    <FlexItem>
      <Title headingLevel={"h1"} style={{ paddingBottom: 20 }}>Wishlist</Title>
      {isLoading ? (
        <Skeleton style={{ height: 200 }} />
      ) : (
        <GalleryView products={products} />
      )}
    </FlexItem>
  );
}
