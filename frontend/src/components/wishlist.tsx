import { GalleryView } from "./Gallery";
import { Title, FlexItem, Skeleton } from "@patternfly/react-core";
import { fetchWishlist } from "../services/products";
import { useQuery } from "@tanstack/react-query";

export function Wishlist() {
  const { data, isError, isLoading } = useQuery({
    queryKey: ["wishlist"], // A unique key for this query
    queryFn: fetchWishlist, // The async function to fetch data
  });

  return (
    <FlexItem>
      <Title headingLevel={"h1"} style={{ paddingBottom: 20 }}>Wishlist</Title>
      {isLoading ? (
        <Skeleton style={{ height: 200 }} />
      ) : (
        isError ? <div>Error fetching wishlist</div> : <GalleryView products={data ?? []} />
      )}
    </FlexItem>
  );
}
