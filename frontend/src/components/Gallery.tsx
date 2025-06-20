import { Gallery, GalleryItem } from "@patternfly/react-core";
import { ProductCard } from "./product-card";
import "./Carousel/carousel.css";

interface ProductData {
  id: number;
  title: string;
  description: string;
  price: string;
  imageUrl: string;
  rating: string;
}

interface ProductDictionary {
  [key: string]: ProductData[];
}

export const GalleryView: React.FunctionComponent<ProductDictionary> = ({
  products,
}) => {
  return (
    <div className="gallery-container">
      <Gallery hasGutter>
        {Object.values(products).map((product, index) => (
          <GalleryItem
            className="cards-container"
            style={{
              marginTop: "15px",
              overflow: "hidden",
            }}
            key={index}
          >
            <ProductCard product={product} index={index} />
          </GalleryItem>
        ))}
      </Gallery>
    </div>
  );
};
