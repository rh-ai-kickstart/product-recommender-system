import Slider from "react-slick";
import { ProductCard } from "../product-card";
import AngleLeftIcon from "@patternfly/react-icons/dist/esm/icons/angle-left-icon";
import AngleRightIcon from "@patternfly/react-icons/dist/esm/icons/angle-right-icon";
import "slick-carousel/slick/slick.css";
import "slick-carousel/slick/slick-theme.css";
import "./carousel.css";
import { Button } from "@patternfly/react-core";
import type { ProductData } from "../../types";

interface ProductDictionary {
  [key: string]: ProductData[];
}

interface ArrowProps {
  onClick?: () => void;
}

export const Carousel: React.FunctionComponent<ProductDictionary> = ({
  products,
}) => {
  function NextArrow(props: ArrowProps) {
    const { onClick } = props;
    return (
      <div className={"slick-arrow slick-next"} onClick={onClick}>
        <Button variant="plain" icon={<AngleRightIcon />}></Button>
      </div>
    );
  }

  function PrevArrow(props: ArrowProps) {
    const { onClick } = props;
    return (
      <div className={"slick-arrow slick-prev"} onClick={onClick}>
        <Button variant="plain" icon={<AngleLeftIcon />}></Button>
      </div>
    );
  }

  const settings = {
    dots: true,
    infinite: false,
    speed: 500,
    slidesToShow: 4,
    slidesToScroll: 4,
    initialSlide: 0,
    responsive: [
      {
        breakpoint: 1024,
        settings: {
          slidesToShow: 3,
          slidesToScroll: 3,
          infinite: true,
          dots: true,
        },
      },
      {
        breakpoint: 600,
        settings: {
          slidesToShow: 2,
          slidesToScroll: 2,
          initialSlide: 2,
        },
      },
      {
        breakpoint: 480,
        settings: {
          slidesToShow: 1,
          slidesToScroll: 1,
        },
      },
    ],
    nextArrow: <NextArrow />,
    prevArrow: <PrevArrow />,
  };

  return (
    <div className="slider-container">
      <Slider {...settings}>
        {Object.values(products).map((product, index) => (
          <div
            className="cards-container"
            key={index}
          >
            <ProductCard product={product} index={index} />
          </div>
        ))}
      </Slider>
    </div>
  );
};
